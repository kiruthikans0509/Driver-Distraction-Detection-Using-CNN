import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import winsound
import threading

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

# Model path configuration - using V3 best model for live inference
DISTRACTION_MODEL_PATH = "models/distraction_cnn_webcam_v3_best.keras"

try:
    print(f"Loading Face Landmarker: models/face_landmarker.task...")
    landmarker = FaceLandmarker.create_from_options(options)
    print(f"Loading Eye Model: models/eye_cnn.keras...")
    eye_model = tf.keras.models.load_model("models/eye_cnn.keras")
    print(f"Loading Distraction Model: {DISTRACTION_MODEL_PATH}...")
    distraction_model = tf.keras.models.load_model(DISTRACTION_MODEL_PATH)
    print("All models loaded successfully!")
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to load models: {e}")
    raise SystemExit(1)

# Audio Alert System (Non-blocking winsound with cooldown)
sound_lock = threading.Lock()

def _play_drowsy_pattern():
    with sound_lock:
        try:
            for _ in range(3):
                winsound.Beep(1800, 120)
                time.sleep(0.04)
        except Exception as e:
            print(f"[Sound Warning] Drowsiness sound error: {e}")

def _play_distraction_pattern():
    with sound_lock:
        try:
            winsound.Beep(850, 200)
            time.sleep(0.04)
            winsound.Beep(650, 300)
        except Exception as e:
            print(f"[Sound Warning] Distraction sound error: {e}")

def trigger_audio_alert(alert_type):
    """Trigger non-blocking audio alert in background daemon thread."""
    target = _play_drowsy_pattern if alert_type == "drowsy" else _play_distraction_pattern
    threading.Thread(target=target, daemon=True).start()

# Audio Alert Cooldown (seconds between repeated sounds while alert remains active)
ALERT_COOLDOWN_SECONDS = 5.0
last_drowsy_sound_time = 0.0
last_distraction_sound_time = 0.0

CLASS_NAMES = ["Closed", "Open"]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.2

# State Farm classes, in the order Keras assigned them (alphabetical: c0, c1, c2...c9)
DISTRACTION_CLASSES = [
    "Safe driving",           # c0
    "Texting - right",        # c1
    "Talking on phone - right",  # c2
    "Texting - left",         # c3
    "Talking on phone - left",   # c4
    "Operating radio",        # c5
    "Drinking",                # c6
    "Reaching behind",        # c7
    "Hair/makeup",             # c8
    "Talking to passenger"    # c9
]
DISTRACTION_CONFIRM_THRESHOLD = 20  # frames before confirming a distraction alert

CLOSED_FRAMES_THRESHOLD = 20
closed_frame_counter = 0
drowsy_alert_active = False

distracted_frame_counter = 0
distraction_alert_active = False
current_distraction_label = "Safe driving"
current_distraction_confidence = 1.0


def get_eye_crop(frame, landmarks, eye_indices, w, h, padding=10):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = max(min(xs) - padding, 0), min(max(xs) + padding, w)
    y_min, y_max = max(min(ys) - padding, 0), min(max(ys) + padding, h)
    return frame[y_min:y_max, x_min:x_max]


def preprocess_eye_crop(eye_crop):
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    resized = cv2.resize(equalized, (64, 64))
    normalized = resized / 255.0
    return normalized.reshape(1, 64, 64, 1)


def predict_eye_state_cnn(eye_crop):
    input_tensor = preprocess_eye_crop(eye_crop)
    prediction = eye_model.predict(input_tensor, verbose=0)[0][0]
    label = CLASS_NAMES[1] if prediction > 0.5 else CLASS_NAMES[0]
    return label, prediction


def calculate_ear(landmarks, eye_indices, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices])
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def predict_distraction(frame):
    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(rgb, axis=0)
    processed = preprocess_input(batch.astype(np.float32))
    preds = distraction_model.predict(processed, verbose=0)[0]
    class_idx = np.argmax(preds)
    return DISTRACTION_CLASSES[class_idx], preds[class_idx], preds


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[CRITICAL ERROR] Could not open webcam (index 0). Please check your camera connection.")
    raise SystemExit(1)

start_time = time.time()
frame_count = 0
DISTRACTION_CHECK_EVERY_N_FRAMES = 5  # run distraction CNN every 5th frame, not every frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Empty frame received from webcam. Exiting loop.")
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # --- Drowsiness (every frame, needs precision) ---
    if result.face_landmarks:
        h, w, _ = frame.shape
        landmarks = result.face_landmarks[0]

        eye_crop = get_eye_crop(frame, landmarks, LEFT_EYE, w, h)
        ear = calculate_ear(landmarks, LEFT_EYE, w, h)

        if eye_crop.size > 0:
            cnn_label, cnn_confidence = predict_eye_state_cnn(eye_crop)
            ear_says_closed = ear < EAR_THRESHOLD

            if cnn_label == "Closed" and ear_says_closed:
                per_frame_state = "Closed"
            elif cnn_label == "Open" and not ear_says_closed:
                per_frame_state = "Open"
            else:
                per_frame_state = "Uncertain"

            if per_frame_state == "Closed":
                closed_frame_counter += 1
            else:
                closed_frame_counter = 0
                drowsy_alert_active = False
                last_drowsy_sound_time = 0.0

            if closed_frame_counter >= CLOSED_FRAMES_THRESHOLD:
                drowsy_alert_active = True
                now = time.time()
                if now - last_drowsy_sound_time >= ALERT_COOLDOWN_SECONDS:
                    trigger_audio_alert("drowsy")
                    last_drowsy_sound_time = now

    # --- Distraction (every Nth frame, CNN is heavier) ---
    if frame_count % DISTRACTION_CHECK_EVERY_N_FRAMES == 0:
        current_distraction_label, current_distraction_confidence, all_preds = predict_distraction(frame)

        # Print full diagnostics approximately once per second (every 4th distraction check)
        if (frame_count // DISTRACTION_CHECK_EVERY_N_FRAMES) % 4 == 0:
            print("=== DISTRACTION DIAGNOSTIC ===")
            for i, (cls_name, prob) in enumerate(zip(DISTRACTION_CLASSES, all_preds * 100)):
                print(f"{cls_name:25} {prob:5.1f}%")
            print(f"Top prediction: {current_distraction_label} ({current_distraction_confidence * 100:.1f}%)")
            print(f"Counter: {distracted_frame_counter}/{DISTRACTION_CONFIRM_THRESHOLD}")
            print("=============")

        if current_distraction_label != "Safe driving":
            distracted_frame_counter += 1
        else:
            distracted_frame_counter = 0
            distraction_alert_active = False
            last_distraction_sound_time = 0.0

        if distracted_frame_counter >= DISTRACTION_CONFIRM_THRESHOLD:
            distraction_alert_active = True
            now = time.time()
            if now - last_distraction_sound_time >= ALERT_COOLDOWN_SECONDS and current_distraction_label != "Safe driving":
                trigger_audio_alert("distraction")
                last_distraction_sound_time = now

    # --- Display everything ---
    drowsy_color = (0, 0, 255) if drowsy_alert_active else (0, 255, 0)
    distract_color = (0, 0, 255) if distraction_alert_active else (0, 255, 0)

    cv2.putText(frame, f"Eyes: {'Closed' if closed_frame_counter > 0 else 'Open'} (streak {closed_frame_counter})",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, drowsy_color, 2)
    cv2.putText(frame, f"Scene: {current_distraction_label} ({current_distraction_confidence*100:.1f}%) [{distracted_frame_counter}/{DISTRACTION_CONFIRM_THRESHOLD}]",
                (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, distract_color, 2)

    if drowsy_alert_active:
        cv2.putText(frame, "DROWSINESS ALERT!", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    if distraction_alert_active:
        cv2.putText(frame, "DISTRACTION ALERT!", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()