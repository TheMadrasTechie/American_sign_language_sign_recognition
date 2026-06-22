"""
Live ASL Alphabet Recognition - Pro UI
Right-hand keypoints (21 * 3 = 63), 30-frame sequences, 26 classes A-Z.
Built for TF 2.15 / Keras 2 (loads the old .h5 natively).

Run:   python live_alphabet_pro.py
Keys:  q = quit | c = clear sentence | SPACE = add space to sentence
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "alphabet_30_30_all_4.h5"
CAMERA_INDEX = 0
SEQUENCE_LENGTH = 30
THRESHOLD = 0.5
MAX_SENTENCE = 12          # letters kept in the banner before trimming

actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                    'Y', 'Z'])

# Colour palette (BGR)
C_BG_PANEL   = (28, 28, 32)
C_ACCENT     = (245, 166, 35)     # amber
C_ACCENT_2   = (88, 214, 141)     # green
C_TEXT       = (240, 240, 240)
C_TEXT_DIM   = (150, 150, 155)
C_BAR        = (245, 166, 35)
C_BAR_TOP    = (88, 214, 141)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return out, results


def draw_hand(image, results):
    """Cleaner skeleton: glowing connections + depth-shaded joints."""
    lm = results.right_hand_landmarks
    if not lm:
        return
    h, w = image.shape[:2]
    pts = [(int(p.x * w), int(p.y * h), p.z) for p in lm.landmark]

    # Connections (draw twice: thick dark underlay, then bright line = glow)
    for a, b in mp_holistic.HAND_CONNECTIONS:
        pa, pb = pts[a][:2], pts[b][:2]
        cv2.line(image, pa, pb, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(image, pa, pb, C_ACCENT, 2, cv2.LINE_AA)

    # Joints, radius scaled by depth (closer = bigger/brighter)
    zs = [p[2] for p in pts]
    zmin, zmax = min(zs), max(zs)
    span = (zmax - zmin) or 1e-6
    for (x, y, z) in pts:
        t = 1.0 - (z - zmin) / span          # 1 = closest
        r = int(3 + 5 * t)
        col = (int(66 + 150 * t), int(180 + 60 * t), int(230))
        cv2.circle(image, (x, y), r + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), r, col, -1, cv2.LINE_AA)


def extract_keypoints(results):
    if results.right_hand_landmarks:
        return np.array([[r.x, r.y, r.z]
                         for r in results.right_hand_landmarks.landmark]).flatten()
    return np.zeros(21 * 3)


# ---------------------------------------------------------------------------
# UI drawing
# ---------------------------------------------------------------------------
def rounded_panel(img, x1, y1, x2, y2, color, alpha=0.78):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_prob_panel(image, res):
    """Right-side panel showing the top predictions as bars."""
    h, w = image.shape[:2]
    pw = 230
    x0 = w - pw
    rounded_panel(image, x0, 0, w, h, C_BG_PANEL, 0.72)

    cv2.putText(image, "CONFIDENCE", (x0 + 16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT_DIM, 1, cv2.LINE_AA)

    # Top 8 predictions, sorted
    order = np.argsort(res)[::-1][:8]
    y = 58
    for rank, idx in enumerate(order):
        prob = float(res[idx])
        label = actions[idx]
        bar_w = int(prob * (pw - 80))
        col = C_BAR_TOP if rank == 0 else C_BAR
        cv2.putText(image, label, (x0 + 16, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    C_TEXT if rank == 0 else C_TEXT_DIM, 2, cv2.LINE_AA)
        cv2.rectangle(image, (x0 + 46, y + 5), (x0 + 46 + (pw - 80), y + 22),
                      (50, 50, 55), -1)
        cv2.rectangle(image, (x0 + 46, y + 5), (x0 + 46 + bar_w, y + 22),
                      col, -1)
        cv2.putText(image, f"{prob*100:4.0f}%", (x0 + pw - 28, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT_DIM, 1, cv2.LINE_AA)
        y += 34


def draw_sentence_bar(image, sentence, top_conf):
    h, w = image.shape[:2]
    rounded_panel(image, 0, 0, w - 230, 56, C_BG_PANEL, 0.78)
    text = ''.join(sentence) if sentence else "..."
    cv2.putText(image, text, (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_TEXT, 2, cv2.LINE_AA)
    # confidence meter strip under banner
    meter_w = int((w - 230) * top_conf)
    cv2.rectangle(image, (0, 56), (w - 230, 60), (50, 50, 55), -1)
    col = C_ACCENT_2 if top_conf > THRESHOLD else C_ACCENT
    cv2.rectangle(image, (0, 56), (meter_w, 60), col, -1)


def draw_status(image, fps, hand_present):
    h, w = image.shape[:2]
    rounded_panel(image, 0, h - 34, 200, h, C_BG_PANEL, 0.7)
    cv2.putText(image, f"FPS {fps:4.1f}", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1, cv2.LINE_AA)
    dot = C_ACCENT_2 if hand_present else (80, 80, 90)
    cv2.circle(image, (130, h - 17), 7, dot, -1, cv2.LINE_AA)
    cv2.putText(image, "HAND", (145, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                C_TEXT if hand_present else C_TEXT_DIM, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model loaded. Starting camera...")

    sequence, sentence, predictions = [], [], []
    res = np.zeros(len(actions))
    prev_t = time.time()
    fps = 0.0

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {CAMERA_INDEX}. Try 0/1/2.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)   # mirror = feels natural

            image, results = mediapipe_detection(frame, holistic)
            hand_present = results.right_hand_landmarks is not None
            draw_hand(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predictions.append(np.argmax(res))

                if len(predictions) >= 10 and \
                        np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > THRESHOLD:
                        letter = actions[np.argmax(res)]
                        if not sentence or letter != sentence[-1]:
                            sentence.append(letter)

                if len(sentence) > MAX_SENTENCE:
                    sentence = sentence[-MAX_SENTENCE:]

            # FPS (smoothed)
            now = time.time()
            inst = 1.0 / max(now - prev_t, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            prev_t = now

            top_conf = float(res[np.argmax(res)]) if res.any() else 0.0
            draw_prob_panel(image, res)
            draw_sentence_bar(image, sentence, top_conf)
            draw_status(image, fps, hand_present)

            cv2.imshow('ASL Alphabet - Pro', image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []
            elif key == ord(' '):
                sentence.append(' ')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()