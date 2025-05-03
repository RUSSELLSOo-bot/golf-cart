#!/usr/bin/env python3
import os
import cv2
import numpy as np

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

def draw_keypoints(frame, kps, thresh=0.3):
    h, w, _ = frame.shape
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


def init_kalman_filters(num_keypoints, dt=1/30, process_noise=1e-2, meas_noise=1e-1):
    """
    Create one 4D→2D constant‐velocity Kalman filter per keypoint.
    State: [x, y, vx, vy];  Measurement: [x, y]
    Returns a list of cv2.KalmanFilter objects of length num_keypoints.
    """
    filters = []
    for _ in range(num_keypoints):
        kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        kf.errorCovPost = np.eye(4, dtype=np.float32)
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




# ─── 3) Main loop ───────────────────────────
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    filters = init_kalman_filters(17)



    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kps = detect_pose(frame)

        kps_smooth = filter_keypoints(kps, filters)

        nosey, nosex, noseConfidence = kps_smooth[7]
        h, w = frame.shape[:2]

        print(f"nose → y={nosey*h:.1f}, x={nosex*w:.1f}, score={noseConfidence:.1f}")

        draw_keypoints(frame, kps_smooth)

        cv2.imshow("MoveNet Lightning (UINT8 TFLite)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
