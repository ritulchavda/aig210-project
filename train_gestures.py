import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

GESTURE_KEYS = {
    ord("1"): "Palm",
    ord("2"): "Index",
    ord("3"): "TwoFinger",
}

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    print(f"Downloading model to {model_path} ...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def extract_feature(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    wrist = coords[0]
    rel = coords - wrist
    norms = np.linalg.norm(rel, axis=1)
    scale = float(np.max(norms))
    if scale < 1e-6:
        return None

    rel /= scale
    return rel.reshape(-1)


def train_profiles(dataset, margin):
    profiles = {}
    for label, samples in dataset.items():
        arr = np.array(samples, dtype=np.float32)
        centroid = np.mean(arr, axis=0)
        dists = np.linalg.norm(arr - centroid, axis=1)

        threshold = float(np.percentile(dists, 92) + margin)
        profiles[label] = {
            "centroid": centroid.tolist(),
            "threshold": threshold,
            "sample_count": int(arr.shape[0]),
        }

    return profiles


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]

    for a, b in HAND_CONNECTIONS:
        p1 = landmarks[a]
        p2 = landmarks[b]
        cv2.line(
            frame,
            (int(p1.x * w), int(p1.y * h)),
            (int(p2.x * w), int(p2.y * h)),
            (110, 220, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    for p in landmarks:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (210, 245, 255), -1, lineType=cv2.LINE_AA)


def draw_help(frame, selected_label, counts, target, ready_labels, start_time):
    h, _ = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (620, 220), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (620, 220), (95, 210, 255), 2)

    lines = [
        "Gesture Trainer (static): 1=Palm, 2=Index, 3=TwoFinger",
        "Choose a gesture key, hold it in frame to collect samples",
        "Press S to save when all gestures reach target",
        "Press Q to quit",
        f"Selected: {selected_label if selected_label else 'None'}",
    ]

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (22, 36 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 240, 255), 1)

    y0 = 142
    for i, label in enumerate(["Palm", "Index", "TwoFinger"]):
        c = counts[label]
        done = "DONE" if label in ready_labels else "..."
        cv2.putText(
            frame,
            f"{label}: {c}/{target} {done}",
            (22, y0 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (130, 230, 255) if label in ready_labels else (190, 210, 230),
            1,
        )

    elapsed = time.time() - start_time
    cv2.putText(
        frame,
        f"Session {elapsed:0.0f}s",
        (22, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 190, 210),
        1,
    )


def main():
    parser = argparse.ArgumentParser(description="Learn hand gesture profiles for website static gesture recognition")
    parser.add_argument("--output", default="gesture_profiles.json", help="Output JSON path")
    parser.add_argument("--samples-per-gesture", type=int, default=160, help="Samples per gesture")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--margin", type=float, default=0.02, help="Distance margin added to threshold")
    parser.add_argument(
        "--model",
        default="hand_landmarker.task",
        help="Path to hand_landmarker.task model file (auto-downloaded if missing)",
    )
    args = parser.parse_args()

    dataset = {"Palm": [], "Index": [], "TwoFinger": []}
    counts = {"Palm": 0, "Index": 0, "TwoFinger": 0}
    selected_label = None
    frame_skip = 0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    model_path = ensure_model(Path(args.model))
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    start_time = time.time()

    with vision.HandLandmarker.create_from_options(options) as hand_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int(time.monotonic() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            ready_labels = {
                label for label in dataset if len(dataset[label]) >= args.samples_per_gesture
            }

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, landmarks)

                if selected_label and counts[selected_label] < args.samples_per_gesture:
                    frame_skip += 1
                    if frame_skip % 2 == 0:
                        feature = extract_feature(landmarks)
                        if feature is not None:
                            dataset[selected_label].append(feature)
                            counts[selected_label] += 1

            if selected_label and counts[selected_label] >= args.samples_per_gesture:
                selected_label = None

            draw_help(frame, selected_label, counts, args.samples_per_gesture, ready_labels, start_time)
            cv2.imshow("Gesture Trainer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in GESTURE_KEYS:
                selected_label = GESTURE_KEYS[key]
            elif key in (ord("q"), 27):
                break
            elif key in (ord("s"), ord("S")):
                if all(counts[g] >= args.samples_per_gesture for g in dataset):
                    break

    cap.release()
    cv2.destroyAllWindows()

    if not all(counts[g] >= args.samples_per_gesture for g in dataset):
        raise RuntimeError("Not enough samples. Train all 3 static gestures before saving.")

    profiles = train_profiles(dataset, margin=args.margin)
    payload = {
        "trained": True,
        "version": 1,
        "created_at": int(time.time()),
        "featureLength": 42,
        "labels": ["Palm", "Index", "TwoFinger"],
        "profiles": profiles,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved learned gesture profiles to {output_path.resolve()}")


if __name__ == "__main__":
    main()
