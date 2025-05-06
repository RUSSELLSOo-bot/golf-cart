#!/usr/bin/env python3
import os
import cv2
import numpy as np
import time

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
    """
    - BGR→RGB
    - Resize to INPUT_SIZE
    - Expand batch dim
    - Cast to uint8 (0–255)
    """
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

    # Draw FPS on the top-left corner
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # joints
    for y, x, score in kps:
        if score < thresh: continue
        px, py = int(x * w), int(y * h)
        cv2.circle(frame, (px, py), 4, (0,255,0), -1)
    # bones
    for (i,j), col in EDGES.items():
        yi, xi, si = kps[i]
        yj, xj, sj = kps[j]
        if si < thresh or sj < thresh: continue
        pt1 = (int(xi*w), int(yi*h))
        pt2 = (int(xj*w), int(yj*h))
        color = {"m":(255,0,255),"c":(255,255,0),"y":(0,255,255)}[col]
        cv2.line(frame, pt1, pt2, color, 2)


def init_kalman_filters(num_keypoints, dt=1/30, process_noise=1e-2, meas_noise=1e-2):
    """
    Create one 4D→2D constant‐velocity Kalman filter per keypoint.
    State: [x, y, vx, vy];  Measurement: [x, y]
    Returns a list of cv2.KalmanFilter objects of length num_keypoints.
    """
    filters = []
    for _ in range(num_keypoints):
        # State: [x, y, vx, vy, ax, ay]; Measurement: [x, y]
        kf = cv2.KalmanFilter(6, 2, 0, cv2.CV_32F)

        # State transition matrix (accounts for acceleration)
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5 * dt**2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], np.float32)

        # Measurement matrix (maps state to measurements)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], np.float32)

        kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        kf.errorCovPost = np.eye(6, dtype=np.float32)
        filters.append(kf)
    return filters


def filter_keypoints(kps, filters):
    """
    Apply your list of Kalman filters to the raw kps array.
    - kps: (17,3) array of (y_norm, x_norm, score)
    - filters: list of length 17 from init_kalman_filters()
    Returns a new (17,3) array of smoothed (y_norm, x_norm, score).
    """
    smoothed = []
    for idx, (y, x, score) in enumerate(kps):
        # Predict + correct
        filters[idx].predict()
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        post = filters[idx].correct(meas)
        sx, sy = float(post[0]), float(post[1])
        smoothed.append((sy, sx, score))
    return np.array(smoothed, dtype=np.float32)


def get_available_cameras(max_cameras=1):
    """
    Detect available camera indices.
    """
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    #returns list of int indeces of cameras found        
    return available_cameras

def main():
    available_cameras = get_available_cameras()
    if not available_cameras:
        print("ERROR: No cameras detected.")
        return

    
    num_cameras = int(len(available_cameras))
    if num_cameras < 1 or num_cameras > len(available_cameras):
        print("Invalid number of cameras.")
        return

    # caps is list of videoCapture objects for all indices in available_cameras
    # filters_list is list of filter objects (but all are the same)
    caps = [cv2.VideoCapture(idx, cv2.CAP_DSHOW) for idx in available_cameras[:num_cameras]]
    filters_list = [init_kalman_filters(17) for _ in range(num_cameras)]

    prev_time = time.time()

    while True:
        frames = []
        keypoints = []
        


        for cap, filters in zip(caps, filters_list):
            ret, frame = cap.read()
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            
            for kf in filters:
                kf.transitionMatrix = np.array([
                    [1, 0, dt, 0, 0.5 * dt**2, 0],
                    [0, 1, 0, dt, 0, 0.5 * dt**2],
                    [0, 0, 1, 0, dt, 0],
                    [0, 0, 0, 1, 0, dt],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ], np.float32)

            kps = detect_pose(frame)
            kps_smooth = filter_keypoints(kps, filters)
            draw_keypoints(frame, kps_smooth, dt)
            frames.append(frame)
            keypoints.append(kps_smooth)

        # Display all frames
        for i, frame in enumerate(frames):
            cv2.imshow(f"Camera {i}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
