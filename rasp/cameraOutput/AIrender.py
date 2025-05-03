#!/usr/bin/env python3
import os
import urllib.request
import numpy as np
import cv2
import tensorflow as tf

# If you're using the full TF package instead of tflite-runtime, uncomment:


try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # fallback to full TF if tflite-runtime isn't installed
    from tensorflow.lite.python.interpreter import Interpreter

# ──────────────── DOWNLOAD THE MODEL IF NEEDED ────────────────
MODEL_NAME = "movenet_lightning.tflite"
MODEL_URL = (
    "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/3?tf-hub-format=compressed"
)

if not os.path.exists(MODEL_NAME):
    print("Downloading MoveNet model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
    print("Downloaded", MODEL_NAME)

# ──────────────── SETUP THE INTERPRETER ────────────────
interpreter = Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# MoveNet Lightning expects a square input (192×192)
INPUT_SIZE = input_details["shape"][1]

# COCO keypoint connections for drawing the skeleton
EDGES = {
    (0,1):  "m", (0,2):  "m",
    (1,3):  "m", (2,4):  "m",
    (0,5):  "c", (0,6):  "c",
    (5,6):  "c", (5,7):  "c",
    (7,9):  "c", (6,8):  "c",
    (8,10): "c", (5,11): "y",
    (6,12): "y", (11,12):"y",
    (11,13):"y",(13,15):"y",
    (12,14):"y",(14,16):"y"
}

def preprocess(frame):
    # Resize and normalize the image to fit the model input size
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    inp = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    return inp

def draw_keypoints(frame, modelPoints, thresh=0.3):
    h, w, _ = frame.shape

    # modelPoints: array of shape (17,3) as (y, x, score)
    for i, (y, x, s) in enumerate(modelPoints):
        if s < thresh:
            continue
        px, py = int(x * w), int(y * h)
        cv2.circle(frame, (px, py), 4, (0,255,0), -1)


    # draw edges
    for (i,j), col in EDGES.items():
        yi, xi, si = modelPoints[i]; yj, xj, sj = modelPoints[j]
        if si > thresh and sj > thresh:
            pt1 = (int(xi*w), int(yi*h))
            pt2 = (int(xj*w), int(yj*h))
            color = {
                "m": (255, 0, 255),  # magenta
                "c": (255,255,0),    # cyan
                "y": (0,255,255)     # yellow
            }[col]
            cv2.line(frame, pt1, pt2, color, 2)


def detect_pose(frame):
    inp = preprocess(frame)
    interpreter.set_tensor(input_details["index"], inp)
    interpreter.invoke()
    modelPoints = interpreter.get_tensor(output_details["index"])[0][0]  # (17,3)
    return modelPoints

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        modelPoints = detect_pose(frame)
        draw_keypoints(frame, modelPoints)

        cv2.imshow("MoveNet Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()