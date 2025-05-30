#!/usr/bin/env python3
import os
import cv2
import numpy as np
import time
from typing import List
from scipy.signal import correlate
from dataclasses import dataclass
from typing import Tuple, List, Optional

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
def set_camera_focus(cap):
    """Helper function to set camera focus settings"""
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap.set(cv2.CAP_PROP_FOCUS, 250)    # Set fixed focus value
    return cap

def init_kalman_filters(num_keypoints, 
            pos_var_x, pos_var_y,           # Position variances
            vel_var_x, vel_var_y,           # Velocity variances
            acc_var_x, acc_var_y,           # Acceleration variances
            dt=1/30):
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
        kf.processNoiseCov = np.eye(6, dtype=np.float32)


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
            cap = set_camera_focus(cap)  # Set focus before testing
            available_cameras.append(i)
            cap.release()
    return available_cameras

def make_const_accel_Q(q: float, dt: float) -> np.ndarray:
        dt2 = dt*dt
        dt3 = dt2*dt
        dt4 = dt3*dt
        dt5 = dt4*dt
        Q = q * np.array([
            [dt5/20,     0,     dt4/8,      0,    dt3/6,    0],
            [0,     dt5/20,      0,     dt4/8,      0,    dt3/6],
            [dt4/8,      0,    dt3/3,      0,    dt2/2,    0],
            [0,     dt4/8,      0,    dt3/3,      0,    dt2/2],
            [dt3/6,      0,    dt2/2,      0,       dt,    0],
            [0,     dt3/6,      0,    dt2/2,      0,      dt],
        ], dtype=np.float32)
        return Q
class NoiseParameters:
    def __init__(self):
        self.process_noise = 1e-10
        self.measurement_scale = 1.0

    def update_filters(self, filters: List[cv2.KalmanFilter], original_meas_cov: np.ndarray, dt: float):
        for kf in filters:
            # dynamic Q
            kf.processNoiseCov = make_const_accel_Q(self.process_noise, dt)
            # scaled measurement noise
            kf.measurementNoiseCov = original_meas_cov * self.measurement_scale

class NoseFilter:
    def __init__(self):
        self.noise_params = NoiseParameters()
        self.kf = None
        self.original_meas_cov = None
        self.prev_pos = None
        self.prev_vel = None
        self.prev_time = time.time()
        self.is_calibrated = False
        self.auto_calibrator = None
    
    def calibrate(self, duration=10):
        """Auto-calibrate filter parameters"""
        self.auto_calibrator = AutoCalibrator(interpreter, input_det, output_det)
        
        # Run two-phase calibration
        still_data, move_data = self.auto_calibrator.run_calibration(duration)
        
        # Calculate measurement noise covariance from still phase
        var_px = np.var(still_data.positions[:,0])
        var_py = np.var(still_data.positions[:,1])
        var_vx = np.var(still_data.velocities[:,0])
        var_vy = np.var(still_data.velocities[:,1])
        var_ax = np.var(still_data.accelerations[:,0])
        var_ay = np.var(still_data.accelerations[:,1])
        
        # Initialize Kalman filter
        self.kf = init_kalman_filters(1, var_px, var_py, var_vx, var_vy, var_ax, var_ay)[0]
        
        # Store original measurement covariance
        self.original_meas_cov = np.diag([
            var_px, var_py, var_vx, var_vy, var_ax, var_ay
        ]).astype(np.float32)
        
        # Find optimal noise parameters
        best_q, best_r = self.auto_calibrator.optimize_parameters(still_data, move_data)
        
        # Set optimal parameters
        self.noise_params.process_noise = best_q
        self.noise_params.measurement_scale = best_r
        
        self.is_calibrated = True
        return self
        
    def update(self, dt: float, x: float, y: float) -> tuple[float, float]:
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
        self.noise_params.update_filters([self.kf], self.original_meas_cov, dt)
        
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

def filter_point(x: float,dt, y: float, filter: NoseFilter) -> tuple[float, float]:
    """Filter a single point using provided calibrated filter"""
    return filter.update(x, y, dt)

def main():
    available_cameras = get_available_cameras()
    if not available_cameras:
        print("ERROR: No cameras detected.")
        return

    num_cameras = int(len(available_cameras))
    if num_cameras < 1:
        print("Invalid number of cameras.")
        return
    print('ok')
    # Single calibration sequence for both parameter optimization and filter initialization
    print("\n=== Running Auto-calibration ===")
    auto_calibrator = AutoCalibrator(interpreter, input_det, output_det)
    still_data, move_data = auto_calibrator.run_calibration(duration_per_phase=10)
    best_q, best_r = auto_calibrator.optimize_parameters(still_data, move_data)
    print(f"Optimal parameters found:")
    print(f"Process noise (Q): {best_q:.2e}")
    print(f"Measurement scale (R): {best_r:.2f}")

    # Use still_data for variance calculations (same as calibrate_measurement_noise)
    var_px = np.var(still_data.positions[:,0])
    var_py = np.var(still_data.positions[:,1])
    var_vx = np.var(still_data.velocities[:,0])
    var_vy = np.var(still_data.velocities[:,1])
    var_ax = np.var(still_data.accelerations[:,0])
    var_ay = np.var(still_data.accelerations[:,1])

    # Remove calibrate_measurement_noise function since we now use auto_calibrator data

    # Initialize measurement covariance
    meas_cov = np.diag([var_px, var_py, var_vx, var_vy, var_ax, var_ay]).astype(np.float32)
    scaled_meas_cov = meas_cov * best_r

    # Setup cameras and filters
    caps = [cv2.VideoCapture(idx, cv2.CAP_DSHOW) for idx in available_cameras[:num_cameras]]
    caps = [set_camera_focus(cap) for cap in caps]
    filters_list = [init_kalman_filters(17, var_px, var_py, var_vx, var_vy, var_ax, var_ay) 
                   for _ in range(num_cameras)]

    # Initialize filters with optimal parameters
    for filters in filters_list:
        for kf in filters:
            kf.measurementNoiseCov = scaled_meas_cov
            # Initial process noise (will be updated with dt)
            kf.processNoiseCov = make_const_accel_Q(best_q, 1/30)

    prev_time = time.time()
    prev_kps = None
    prev_vels = None

    while True:
        frames = []
        keypoints = []
        
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        # Update process noise with current dt for all filters
        process_noise = make_const_accel_Q(best_q, dt)
        for filters in filters_list:
            for kf in filters:
                kf.processNoiseCov = process_noise

        for cap, filters in zip(caps, filters_list):
            ret, frame = cap.read()

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
                    kf.predict() #this step executes ajdacent processnoise(xrawBEFORE +vbefroe * dt + 1/2 * abefore *dt**2)
                    #and stores it in the kf object
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    cv2.destroyWindow('Noise Parameters')

@dataclass
class CalibrationData:
    positions: np.ndarray    # Shape: (N,2) for x,y positions
    velocities: np.ndarray   # Shape: (N-1,2) for vx,vy
    accelerations: np.ndarray # Shape: (N-2,2) for ax,ay
    timestamps: np.ndarray   # Shape: (N,) for time points
    
class AutoCalibrator:
    def __init__(self, interpreter, input_det, output_det):
        self.interpreter = interpreter
        self.input_det = input_det
        self.output_det = output_det
        # Calculate base measurement covariance in run_calibration
        self.original_meas_cov = None
    
    def run_calibration(self, duration_per_phase=10) -> Tuple[CalibrationData, CalibrationData]:
        """Run two-phase calibration process"""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        cap = set_camera_focus(cap)
        
        try:
            # Phase 1: Still calibration
            still_data = self._run_calibration_phase(cap, duration_per_phase, "STILL")
            time.sleep(2)  # Brief pause between phases
            # Phase 2: Movement calibration  
            move_data = self._run_calibration_phase(cap, duration_per_phase, "MOVEMENT")
            return still_data, move_data
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _run_calibration_phase(self, cap, duration: int, phase: str) -> CalibrationData:
        """Run single calibration phase with UI"""
        # Show countdown
        for i in range(5,0,-1):
            ret, frame = cap.read()
            if ret:
                self._draw_countdown(frame, f"Get ready for {phase} phase", i)
                cv2.imshow("Calibration", frame)
                cv2.waitKey(1000)  # 1 second delay
                
        # Collect data
        positions = []
        velocities = []
        accelerations = []
        timestamps = []
        
        start_time = time.time()
        prev_pos = None
        prev_vel = None
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret: continue
            
            current_time = time.time() - start_time
            kps = detect_pose(frame)
            y, x, score = kps[0]  # Nose keypoint
            
            if score > 0.3:  # Confidence threshold
                current_pos = np.array([x, y])
                positions.append(current_pos)
                timestamps.append(current_time)
                
                if prev_pos is not None:
                    dt = timestamps[-1] - timestamps[-2]
                    current_vel = (current_pos - prev_pos) / dt
                    velocities.append(current_vel)
                    
                    if prev_vel is not None:
                        current_acc = (current_vel - prev_vel) / dt
                        accelerations.append(current_acc)
                    
                    prev_vel = current_vel
                else:
                    prev_vel = np.zeros(2, dtype=np.float32)
                prev_pos = current_pos
                
                # Show progress
                progress = (time.time() - start_time) / duration * 100
                self._draw_progress(frame, f"{phase} Calibration", progress)
                cv2.circle(frame, (int(x*frame.shape[1]), int(y*frame.shape[0])), 
                          6, (0,0,255), -1)
                cv2.imshow("Calibration", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        return CalibrationData(
            np.array(positions, dtype=np.float32),  # Ensure float32
            np.array(velocities, dtype=np.float32),
            np.array(accelerations, dtype=np.float32),
            np.array(timestamps, dtype=np.float32)
        )
    
    @staticmethod
    def _calculate_jitter(filtered: np.ndarray, raw: np.ndarray) -> float:
        """
        Calculate jitter as direct std comparison between filtered and raw data.
        Lower filtered std means better noise reduction.
        """
        raw_std = np.std(raw, axis=0)      # Standard deviation of raw data
        filtered_std = np.std(filtered, axis=0)  # Standard deviation of filtered data
        
        # Compare stds directly (no ratio)
        return np.mean(filtered_std)  # Return average std across x,y dimensions

    @staticmethod
    def _calculate_lag(filtered: np.ndarray, raw: np.ndarray) -> float:
        """
        Calculate point-by-point residual variance between filtered and raw signals.
        Smaller residuals mean better tracking.
        """
        # Calculate residuals between corresponding points
        residuals = filtered - raw
        
        # Calculate mean squared error
        mse = np.mean(np.square(residuals))
        return mse
    
    def optimize_parameters(self, still_data: CalibrationData, 
                          move_data: CalibrationData) -> Tuple[float, float]:
        """Find optimal Q,R parameters using grid search"""
        # Calculate base measurement covariance from still data
        var_px = np.var(still_data.positions[:,0])
        var_py = np.var(still_data.positions[:,1])
        var_vx = np.var(still_data.velocities[:,0])
        var_vy = np.var(still_data.velocities[:,1])
        var_ax = np.var(still_data.accelerations[:,0])
        var_ay = np.var(still_data.accelerations[:,1])
        
        # Initialize measurement covariance matrix
        self.original_meas_cov = np.diag([
            var_px, var_py, var_vx, var_vy, var_ax, var_ay
        ]).astype(np.float32)

        # Parameter search space
        qs = np.logspace(-12, -2, 15)  # Process noise values
        rs = np.linspace(0.1, 10.0, 15) # Measurement noise scales
        
        best_cost = float('inf')
        best_q = None
        best_r = None
        
        for q in qs:
            for r in rs:
                # Test parameters on still data (jitter)
                still_filtered = self._simulate_filter(
                    still_data.positions, 
                    still_data.timestamps, q, r
                )
                jitter = self._calculate_jitter(still_filtered, still_data.positions)
                
                # Test parameters on movement data (residual variance)
                move_filtered = self._simulate_filter(
                    move_data.positions,
                    move_data.timestamps, q, r
                )
                residual = self._calculate_lag(move_filtered, move_data.positions)
                
                # Combined cost (weighted sum)
                # Penalize both high jitter (>1) and high residuals
                cost = 0.7 * abs(jitter - 1.0) + 0.3 * residual
                
                if cost < best_cost:
                    best_cost = cost
                    best_q = q
                    best_r = r
                    
        return best_q, best_r
    
    def _simulate_filter(self, positions: np.ndarray, timestamps: np.ndarray, 
                        q: float, r: float) -> np.ndarray:
        """Simulate Kalman filter with given parameters on position data"""
        # Validate input shapes
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"Expected positions shape (N,2), got {positions.shape}")
        if timestamps.ndim != 1:
            raise ValueError(f"Expected timestamps shape (N,), got {timestamps.shape}")
        if len(timestamps) != len(positions):
            raise ValueError("Timestamps and positions must have same length")
        

        q = np.float32(q)
        r = np.float32(r)
        
        # Convert to float32
        positions = positions.astype(np.float32)
        timestamps = timestamps.astype(np.float32)
    
        # Initialize Kalman filter
        kf = cv2.KalmanFilter(6, 6, 0, cv2.CV_32F)
        
        # Initialize matrices (all float32)
        kf.transitionMatrix = np.eye(6, dtype=np.float32)
        kf.measurementMatrix = np.eye(6, dtype=np.float32)
        kf.processNoiseCov = np.eye(6, dtype=np.float32)
        kf.measurementNoiseCov = np.eye(6, dtype=np.float32)
        kf.errorCovPost = np.eye(6, dtype=np.float32)
        kf.errorCovPre = np.eye(6, dtype=np.float32)
        
        # Initial state
        initial_state = np.array([
            positions[0,0],  # x
            positions[0,1],  # y
            0.0,            # vx
            0.0,            # vy
            0.0,            # ax
            0.0             # ay
        ], dtype=np.float32).reshape(6, 1)
        kf.statePost = initial_state
        
        filtered = np.zeros_like(positions)
        filtered[0] = positions[0]  # First point is unfiltered
        prev_pos = positions[0]
        prev_vel = np.zeros(2, dtype=np.float32)

        for i in range(len(positions)):
            dt = timestamps[i] - timestamps[i-1] if i > 0 else 1/30
            
            # Update transition matrix
            kf.transitionMatrix = np.array([
                [1,0, dt,0, 0.5*dt*dt,0],
                [0,1, 0,dt, 0,0.5*dt*dt],
                [0,0, 1,0,  dt,0],
                [0,0, 0,1,  0,dt],
                [0,0, 0,0,  1,0],
                [0,0, 0,0,  0,1]
            ], dtype=np.float32)
            
            # Update noise matrices
            kf.processNoiseCov = make_const_accel_Q(q, dt)
            kf.measurementNoiseCov = self.original_meas_cov * r
            
            # Calculate derivatives
            current_pos = positions[i].astype(np.float32)
            current_vel = ((current_pos - prev_pos) / dt).astype(np.float32) if i > 0 else np.zeros(2, dtype=np.float32)
            current_acc = ((current_vel - prev_vel) / dt).astype(np.float32) if i > 0 else np.zeros(2, dtype=np.float32)
            
            # Predict step
            kf.predict()
            
            # Measurement vector
            z = np.array([
                current_pos[0], current_pos[1],
                current_vel[0], current_vel[1],
                current_acc[0], current_acc[1]
            ], dtype=np.float32).reshape(-1, 1)
            
            # Correct step
            filtered_state = kf.correct(z)
            filtered[i] = [filtered_state[0][0], filtered_state[1][0]]
            
            # Update previous values
            prev_pos = current_pos
            prev_vel = current_vel
            
        return filtered

    # Add to AutoCalibrator class
    def _draw_countdown(self, frame, text: str, seconds: int):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{text}: {seconds}", (w//4, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    def _draw_progress(self, frame, text: str, progress: float):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{text}: {progress:.1f}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

if __name__ == "__main__":
    main()