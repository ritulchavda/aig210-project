import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import seaborn as sns
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

GESTURE_KEYS = {
    ord("1"): "Palm",
    ord("2"): "Index",
    ord("3"): "TwoFinger",
}

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

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def predict_gesture(feature, profiles):
    best_label = "Unknown"
    best_distance = float('inf')
    best_threshold = float('inf')

    # Find closest centroid
    for label, profile in profiles.items():
        centroid = np.array(profile["centroid"])
        threshold = profile["threshold"]
        
        # We also mirror the feature to match JS logic
        mirrored = feature.copy()
        mirrored[::2] *= -1  # negate X coordinates
        
        d1 = euclidean_distance(feature, centroid)
        d2 = euclidean_distance(mirrored, centroid)
        dist = min(d1, d2)

        if dist < best_distance:
            best_distance = dist
            best_label = label
            best_threshold = threshold

    # Apply adaptive threshold (same as JS logic)
    if best_distance <= (best_threshold * 1.5):
        return best_label
    return "Unknown"

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate gesture profiles")
    parser.add_argument("--profiles", default="gesture_profiles.json", help="Path to gesture_profiles.json")
    parser.add_argument("--model", default="hand_landmarker.task", help="Path to hand_landmarker.task")
    parser.add_argument("--samples", type=int, default=50, help="Test samples to collect per gesture")
    args = parser.parse_args()

    # Load trained profiles
    if not Path(args.profiles).is_file():
        print(f"Error: {args.profiles} not found. Please run train_gestures.py first.")
        return

    with open(args.profiles, 'r') as f:
        data = json.load(f)
        profiles = data["profiles"]
        labels = data["labels"]

    print(f"Loaded profiles for: {', '.join(labels)}")

    # Initialize Camera and MediaPipe
    cap = cv2.VideoCapture(0)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    y_true = []
    y_pred = []
    counts = {label: 0 for label in labels}
    selected_label = None

    print("\n--- EVALUATION MODE ---")
    print("Press 1 for Palm, 2 for Index, 3 for TwoFinger to start collecting test samples.")
    print("Press Q to quit early and see results.")

    with vision.HandLandmarker.create_from_options(options) as hand_landmarker:
        while True:
            ok, frame = cap.read()
            if not ok: break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = hand_landmarker.detect_for_video(mp_image, int(time.monotonic() * 1000))

            pred_text = "None"
            
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                feature = extract_feature(landmarks)
                
                if feature is not None:
                    # Make a prediction using the JS equivalent logic
                    prediction = predict_gesture(feature, profiles)
                    pred_text = prediction

                    if selected_label and counts[selected_label] < args.samples:
                        # Only collect every few frames to simulate real variance
                        y_true.append(selected_label)
                        y_pred.append(prediction)
                        counts[selected_label] += 1
                        
                        if counts[selected_label] >= args.samples:
                            print(f"\nCompleted collecting {args.samples} test samples for {selected_label}!")
                            selected_label = None

            # Draw UI
            cv2.putText(frame, f"Evaluating: {selected_label if selected_label else 'Select Key (1,2,3)'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Prediction: {pred_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if pred_text == selected_label else (0, 0, 255), 2)
            
            y_offset = 90
            for k in labels:
                cv2.putText(frame, f"{k}: {counts[k]}/{args.samples}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 25

            cv2.imshow("Evaluation - Gesture Accuracy", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in GESTURE_KEYS:
                selected_label = GESTURE_KEYS[key]
                print(f"Collecting test data for {selected_label}...")
            elif key == ord('q'):
                break

            # Auto-quit if all classes are filled
            if all(c >= args.samples for c in counts.values()):
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(y_true) == 0:
        print("No test data collected.")
        return

    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    print("\n--- RESULTS ---")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels + ["Unknown"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels+["Unknown"], 
                yticklabels=labels+["Unknown"])
    plt.title('Hand Gesture Classification Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    out_file = "confusion_matrix.png"
    plt.savefig(out_file)
    print(f"\nConfusion matrix plot saved to {out_file}")
    plt.show()

if __name__ == "__main__":
    evaluate()
