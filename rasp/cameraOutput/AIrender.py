#!/usr/bin/env python3
import os
import cv2
import numpy as np
import time
from typing import List

# Try standalone tflite-runtime first, else TF fallback
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ─── 1) Load TFLite model ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "movenet_lightning_fp16.tflite")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH!r}")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_det  = interpreter.get_input_details()[0]
output_det = interpreter.get_output_details()[0]

# Confirm we expect UINT8 inputs
assert input_det["dtype"] == np.uint8, (
    f"Expected quantized UINT8 model, got input dtype {input_det['dtype']}"
)

INPUT_SIZE = input_det["shape"][1]  # usually 192×192

# COCO skeleton edges
EDGES = {
    (0,1):"m",(0,2):"m",(1,3):"m",(2,4):"m",
    (0,5):"c",(0,6):"c",(5,6):"c",(5,7):"c",
    (7,9):"c",(6,8):"c",(8,10):"c",(5,11):"y",
    (6,12):"y",(11,12):"y",(11,13):"y",(13,15):"y",
    (12,14):"y",(14,16):"y"
}

# ─── 2) Pre/post processing ──────────────────
def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    return np.expand_dims(rgb, axis=0).astype(np.uint8)

def detect_pose(frame):
    inp = preprocess(frame)
    interpreter.set_tensor(input_det["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_det["index"])  # (1,1,17,3)
    return out[0,0]                                     # → (17,3)

def draw_keypoints(frame, kps, dt, thresh=0.3):
    h, w, _ = frame.shape
    fps = 1 / dt if dt > 0 else 0
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for y, x, score in kps:
        if score < thresh: continue
        px, py = int(x * w), int(y * h)
        cv2.circle(frame, (px, py), 4, (0,255,0), -1)
    for (i,j), col in EDGES.items():
        yi, xi, si = kps[i]
        yj, xj, sj = kps[j]
        if si < thresh or sj < thresh: continue
        pt1 = (int(xi*w), int(yi*h))
        pt2 = (int(xj*w), int(yj*h))
        color = {"m":(255,0,255),"c":(255,255,0),"y":(0,255,255)}[col]
        cv2.line(frame, pt1, pt2, color, 2)

# --- Calibration Step ---
def calibrate_measurement_noise(interpreter, input_det, output_det, cap, duration=10, keypoint_idx=0):
    """
    Calibrate measurement noise for position, velocity, and acceleration.
    Calculates velocity and acceleration from frame-to-frame measurements.
    Returns variances for position, velocity, and acceleration.
    """
    positions = []
    velocities = []
    accelerations = []
    prev_pos = None
    prev_vel = None
    prev_time = time.time()
    start_time = time.time()
    
    print(f"Calibration started: Please stand still for {duration} seconds...")

    while time.time() - start_time < duration:
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        ret, frame = cap.read()
        if not ret:
            continue

        # Get current position
        inp = preprocess(frame)
        interpreter.set_tensor(input_det["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_det["index"])
        kps = out[0, 0]
        y, x, score = kps[keypoint_idx]
        current_pos = np.array([x, y])
        
        # Calculate velocity and acceleration if we have previous positions
        if prev_pos is not None:
            current_vel = (current_pos - prev_pos) / dt
            velocities.append(current_vel)
            
            if prev_vel is not None:
                current_acc = (current_vel - prev_vel) / dt
                accelerations.append(current_acc)
            
            prev_vel = current_vel
        
        positions.append(current_pos)
        prev_pos = current_pos

        # Visual feedback
        cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 6, (0, 0, 255), -1)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
    # Convert to numpy arrays
    positions = np.array(positions)
    velocities = np.array(velocities) if velocities else np.zeros((0, 2))
    accelerations = np.array(accelerations) if accelerations else np.zeros((0, 2))
    
    # Calculate variances
    pos_var_x = np.var(positions[:, 0])
    pos_var_y = np.var(positions[:, 1])
    vel_var_x = np.var(velocities[:, 0]) if len(velocities) > 0 else 0
    vel_var_y = np.var(velocities[:, 1]) if len(velocities) > 0 else 0
    acc_var_x = np.var(accelerations[:, 0]) if len(accelerations) > 0 else 0
    acc_var_y = np.var(accelerations[:, 1]) if len(accelerations) > 0 else 0
    
    # Calculate mean positions
    avg_pos_x = np.mean(positions[:, 0])
    avg_pos_y = np.mean(positions[:, 1])
    
    print(f"Calibration complete:")
    print(f"Position variance: x={pos_var_x:.6f}, y={pos_var_y:.6f}")
    print(f"Velocity variance: x={vel_var_x:.6f}, y={vel_var_y:.6f}")
    print(f"Acceleration variance: x={acc_var_x:.6f}, y={acc_var_y:.6f}")
    
    return (pos_var_x, pos_var_y,           # Position variances
            vel_var_x, vel_var_y,           # Velocity variances
            acc_var_x, acc_var_y,           # Acceleration variances
            avg_pos_x, avg_pos_y)           # Mean positions

def init_kalman_filters(num_keypoints, 
            pos_var_x, pos_var_y,           # Position variances
            vel_var_x, vel_var_y,           # Velocity variances
            acc_var_x, acc_var_y,           # Acceleration variances
            dt=1/30, process_noise=1e-5):
    """
    Create one 6D→2D constant-acceleration Kalman filter per keypoint.
    State: [x, y, vx, vy, ax, ay]; Measurement: [x, y, vx, vy, ax, ay]
    Returns a list of cv2.KalmanFilter objects of length num_keypoints.
    """
    filters = []
    for idx in range(num_keypoints):
        kf = cv2.KalmanFilter(6, 6, 0, cv2.CV_32F)
        kf.transitionMatrix = np.array([
            [1,0, dt,0, 0.5*dt*dt, 0],
            [0,1, 0, dt, 0,        0.5*dt*dt],
            [0,0, 1,  0, dt,       0],
            [0,0, 0,  1, 0,        dt],
            [0,0, 0,  0, 1,        0],
            [0,0, 0,  0, 0,        1],
        ], np.float32)
        kf.measurementMatrix = np.eye(6, dtype=np.float32)
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise


        kf.measurementNoiseCov = np.diag([
            pos_var_x, pos_var_y,
            vel_var_x, vel_var_y,
            acc_var_x, acc_var_y
        ]).astype(np.float32)
        kf.errorCovPost = np.eye(6, dtype=np.float32)
        filters.append(kf)
    return filters

def filter_keypoints(kps, filters):
    smoothed = []
    for idx, (y, x, score) in enumerate(kps):
        filters[idx].predict()
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        post = filters[idx].correct(meas)
        sx, sy = float(post[0]), float(post[1])
        smoothed.append((sy, sx, score))
    return np.array(smoothed, dtype=np.float32)

def get_available_cameras(max_cameras=1):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

class NoiseParameters:
    def __init__(self):
        self.process_noise = 1e-10
        self.measurement_scale = 1.0
        
    def create_window(self):
        cv2.namedWindow('Noise Parameters', cv2.WINDOW_NORMAL)  # Make window resizable
        # Create trackbars for noise adjustment
        cv2.createTrackbar('Process Noise (e^x)', 'Noise Parameters', 50, 100, self._on_process_noise_change)
        cv2.createTrackbar('Measurement Scale', 'Noise Parameters', 10, 100, self._on_measurement_scale_change)
    
    def _on_process_noise_change(self, value):
        # Convert slider value to exponential scale from 1e-5 to 1e5
        self.process_noise = 10 ** ((value - 50) / 10)
        
    def _on_measurement_scale_change(self, value):
        # Convert slider value to 0.1 - 10.0 range
        self.measurement_scale = value / 10.0

    def update_filters(self, filters: List[cv2.KalmanFilter], original_meas_cov: np.ndarray):
        """Update all filters with current noise parameters"""
        for f in filters:
            # Update process noise
            process_noise_mat = np.eye(6, dtype=np.float32) * self.process_noise
            f.processNoiseCov[...] = process_noise_mat
            
            # Update measurement noise
            meas_noise_mat = original_meas_cov * self.measurement_scale
            f.measurementNoiseCov[...] = meas_noise_mat

class NoseFilter:
    def __init__(self):
        self.noise_params = NoiseParameters()
        self.kf = None
        self.original_meas_cov = None
        self.prev_pos = None
        self.prev_vel = None
        self.prev_time = time.time()
        self.is_calibrated = False
        
    def calibrate(self, duration=10):
        """Calibrate the filter using camera feed"""
        print("Starting calibration. Please stay still...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
            
        try:
            var_px, var_py, var_vx, var_vy, var_ax, var_ay, _, _ = calibrate_measurement_noise(
                interpreter, input_det, output_det, cap, duration, keypoint_idx=0
            )
            
            # Initialize Kalman filter with calibrated values
            self.kf = init_kalman_filters(1, var_px, var_py, var_vx, var_vy, var_ax, var_ay)[0]
            
            # Store original measurement covariance
            self.original_meas_cov = np.diag([
                var_px, var_py, var_vx, var_vy, var_ax, var_ay
            ]).astype(np.float32)
            
            # Create noise parameter window
            self.noise_params.create_window()
            self.is_calibrated = True
            
        finally:
            cap.release()
        
        return self
        
    def update(self, x: float, y: float) -> tuple[float, float]:
        """Update filter with new measurement and return filtered position"""
        if not self.is_calibrated:
            raise RuntimeError("Filter must be calibrated before use")
            
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        # Calculate derivatives
        current_pos = np.array([x, y])
        if self.prev_pos is not None:
            current_vel = (current_pos - self.prev_pos) / dt
            current_acc = (current_vel - self.prev_vel) / dt if self.prev_vel is not None else np.zeros(2)
            self.prev_vel = current_vel
        else:
            current_vel = np.zeros(2)
            current_acc = np.zeros(2)
        
        self.prev_pos = current_pos
        
        # Update noise parameters
        self.noise_params.update_filters([self.kf], self.original_meas_cov)
        
        # Update Kalman filter
        self.kf.predict()
        measurement = np.array([
            x, y,
            current_vel[0], current_vel[1],
            current_acc[0], current_acc[1]
        ], dtype=np.float32).reshape(-1, 1)
        
        filtered_state = self.kf.correct(measurement)
        
        return float(filtered_state[0][0]), float(filtered_state[1][0])

def create_calibrated_filter() -> NoseFilter:
    """Create and calibrate a new nose filter"""
    filter = NoseFilter()
    return filter.calibrate()

def filter_point(x: float, y: float, filter: NoseFilter) -> tuple[float, float]:
    """Filter a single point using provided calibrated filter"""
    return filter.update(x, y)

def main():
    available_cameras = get_available_cameras()
    if not available_cameras:
        print("ERROR: No cameras detected.")
        return

    num_cameras = int(len(available_cameras))
    if num_cameras < 1 or num_cameras > len(available_cameras):
        print("Invalid number of cameras.")
        return

    # --- Calibration step for measurement noise ---
    cap_for_calib = cv2.VideoCapture(available_cameras[0], cv2.CAP_DSHOW)
    print("Measurement noise covariance BEFORE calibration (nose):", 1.0)
    var_px, var_py, var_vx, var_vy,var_ax, var_ay, avg_pos_x, avg_pose_y = calibrate_measurement_noise(interpreter, input_det, output_det, cap_for_calib, duration=10, keypoint_idx=0)
    cap_for_calib.release()
    print("Measurement noise covariance AFTER calibration (nose):", var_px, var_py, var_vx, var_vy,var_ax, var_ay, avg_pos_x, avg_pose_y )

    # Use calibrated values for the nose keypoint
    caps = [cv2.VideoCapture(idx, cv2.CAP_DSHOW) for idx in available_cameras[:num_cameras]]
    filters_list = [init_kalman_filters(17, 
                                        pos_var_x=var_px, 
                                        pos_var_y=var_py,
                                        vel_var_x=var_vx,
                                        vel_var_y=var_vy,
                                        acc_var_x=var_ax,
                                        acc_var_y=var_ay, 
                                       ) for _ in range(num_cameras)]

    # Initialize noise parameters
    noise_params = NoiseParameters()
    noise_params.create_window()
    
    # Store original measurement noise covariance
    original_meas_cov = np.diag([
        var_px, var_py,
        var_vx, var_vy,
        var_ax, var_ay
    ]).astype(np.float32)

    prev_time = time.time()
    prev_kps   = None   # last [y,x] positions
    prev_vels  = None   # last [vx,vy] velocities

    while True:
        frames = []
        keypoints = []

        # Update filter parameters and verify changes
        for filters in filters_list:
            noise_params.update_filters(filters, original_meas_cov)
            # Debug print - uncomment to verify values are changing
            # print(f"\rProcess Noise: {filters[0].processNoiseCov[0,0]:.2e}, "
            #       f"Measurement Scale: {filters[0].measurementNoiseCov[0,0]/original_meas_cov[0,0]:.2f}", 
            #       end="")

        for cap, filters in zip(caps, filters_list):
            ret, frame = cap.read()
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time 

            kps = detect_pose(frame)

            if prev_kps is None:
                prev_kps = kps.copy()
                prev_vels = np.zeros((len(kps), 2), dtype=np.float32)
                vels = np.zeros((len(kps), 2), dtype=np.float32)
                accs = np.zeros((len(kps), 2), dtype=np.float32)
                smoothed = []
                for idx, kf in enumerate(filters):
                    x_raw, y_raw = kps[idx,1], kps[idx,0]
                    vx_raw, vy_raw = vels[idx]
                    ax_raw, ay_raw = accs[idx]
                    kf.predict()
                    z = np.array([[x_raw],[y_raw],[vx_raw],[vy_raw],[ax_raw],[ay_raw]]).astype(np.float32)
                    post = kf.correct(z)
                    x_f, y_f = float(post[0]), float(post[1])
                    smoothed.append((x_raw, y_raw, kps[idx,2]))
                    # smoothed.append((x_f, y_f, kps[idx,2]))
                kps_smooth = np.array(smoothed, dtype=np.float32)
            else:
                vels = (kps[:,:2] - prev_kps[:,:2]) / dt
                accs = (vels - prev_vels) / dt
                smoothed = []
                for idx, kf in enumerate(filters):
                    x_raw, y_raw = kps[idx,0], kps[idx,1]
                    vx_raw, vy_raw = vels[idx]
                    ax_raw, ay_raw = accs[idx]
                    kf.predict()
                    z = np.array([[x_raw],[y_raw],[vx_raw],[vy_raw],[ax_raw],[ay_raw]]).astype(np.float32)
                    post = kf.correct(z)
                    x_f, y_f = float(post[0]), float(post[1])
                    smoothed.append((x_f, y_f, kps[idx,2]))
                kps_smooth = np.array(smoothed, dtype=np.float32)
                prev_kps  = kps.copy()
                prev_vels = vels.copy()

            draw_keypoints(frame, kps_smooth, dt)
            frames.append(frame)
            keypoints.append(kps_smooth)

        for i, frame in enumerate(frames):
            cv2.imshow(f"Camera {i}", frame)

            # Add these lines to show current parameter values
            cv2.putText(frame, 
                       f"Process Noise: {noise_params.process_noise:.1e}", 
                       (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (0, 255, 0), 
                       1)
            cv2.putText(frame, 
                       f"Measurement Scale: {noise_params.measurement_scale:.1f}", 
                       (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (0, 255, 0), 
                       1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    cv2.destroyWindow('Noise Parameters')

if __name__ == "__main__":
    main()
