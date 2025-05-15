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

input_det = interpreter.get_input_details()[0]
output_det = interpreter.get_output_details()[0]

# Confirm we expect UINT8 inputs
assert input_det["dtype"] == np.uint8, (
    f"Expected quantized UINT8 model, got input dtype {input_det['dtype']}"
)

INPUT_SIZE = input_det["shape"][1]  # usually 192×192

# COCO skeleton edges
EDGES = {
    (0, 1): "m", (0, 2): "m", (1, 3): "m", (2, 4): "m",
    (0, 5): "c", (0, 6): "c", (5, 6): "c", (5, 7): "c",
    (7, 9): "c", (6, 8): "c", (8, 10): "c", (5, 11): "y",
    (6, 12): "y", (11, 12): "y", (11, 13): "y", (13, 15): "y",
    (12, 14): "y", (14, 16): "y"
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
    return out[0, 0]  # → (17,3)

def draw_keypoints(frame, kps, dt, thresh=0.3):
    h, w, _ = frame.shape

    fps = 1 / dt if dt > 0 else 0

    # Draw FPS on the top-left corner
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw joints
    for y, x, score in kps:
        if score < thresh:
            continue
        px, py = int(x * w), int(y * h)
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    # Draw bones
    for (i, j), col in EDGES.items():
        yi, xi, si = kps[i]
        yj, xj, sj = kps[j]
        if si < thresh or sj < thresh:
            continue
        pt1 = (int(xi * w), int(yi * h))
        pt2 = (int(xj * w), int(yj * h))
        color = {"m": (255, 0, 255), "c": (255, 255, 0), "y": (0, 255, 255)}[col]
        cv2.line(frame, pt1, pt2, color, 2)

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

    # Returns list of int indices of cameras found
    return available_cameras

def main():
    available_cameras = get_available_cameras()
    if not available_cameras:
        print("ERROR: No cameras detected.")
        return

    num_cameras = len(available_cameras)
    caps = [cv2.VideoCapture(idx, cv2.CAP_DSHOW) for idx in available_cameras]

    prev_time = time.time()

    while True:
        frames = []
        keypoints = []
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read from camera.")
                break

            kps = detect_pose(frame)
            # Get nose coordinates (first keypoint)
            if kps is not None and len(kps) > 0:
                y, x, conf = kps[0]  # Nose keypoint
                if conf > 0.3:  # Only print if confidence is high enough
                    h, w, _ = frame.shape
                    pixel_x, pixel_y = int(x * w), int(y * h)
                    print(f"\rNose position - Pixel(x,y): ({pixel_x}, {pixel_y}), Normalized(x,y): ({x:.3f}, {y:.3f})", end="")

            draw_keypoints(frame, kps, dt)
            frames.append(frame)
            keypoints.append(kps)

        # Display all frames
        for i, frame in enumerate(frames):
            cv2.imshow(f"Camera {i}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\nExiting...")
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()