"""
Live application: pro UI built on top of the headless Recognizer.

Public entry point: signspell.run(...) — also wired to the `signspell` CLI.
"""

import time
from typing import Optional

import numpy as np

from .config import ACTIONS, SEQUENCE_LENGTH, DEFAULT_THRESHOLD, STABILITY_WINDOW
from .recognizer import Recognizer

# Colour palette (BGR)
C_BG_PANEL = (28, 28, 32)
C_ACCENT = (245, 166, 35)
C_ACCENT_2 = (88, 214, 141)
C_TEXT = (240, 240, 240)
C_TEXT_DIM = (150, 150, 155)


def _rounded_panel(img, x1, y1, x2, y2, color, alpha=0.78):
    import cv2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_hand(image, results):
    import cv2
    from mediapipe.python.solutions import holistic as mp_holistic

    lm = results.right_hand_landmarks
    if not lm:
        return
    h, w = image.shape[:2]
    pts = [(int(p.x * w), int(p.y * h), p.z) for p in lm.landmark]

    for a, b in mp_holistic.HAND_CONNECTIONS:
        pa, pb = pts[a][:2], pts[b][:2]
        cv2.line(image, pa, pb, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(image, pa, pb, C_ACCENT, 2, cv2.LINE_AA)

    zs = [p[2] for p in pts]
    zmin, zmax = min(zs), max(zs)
    span = (zmax - zmin) or 1e-6
    for (x, y, z) in pts:
        t = 1.0 - (z - zmin) / span
        r = int(3 + 5 * t)
        col = (int(66 + 150 * t), int(180 + 60 * t), 230)
        cv2.circle(image, (x, y), r + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), r, col, -1, cv2.LINE_AA)


def _draw_prob_panel(image, probs):
    import cv2
    h, w = image.shape[:2]
    pw = 230
    x0 = w - pw
    _rounded_panel(image, x0, 0, w, h, C_BG_PANEL, 0.72)
    cv2.putText(image, "CONFIDENCE", (x0 + 16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT_DIM, 1, cv2.LINE_AA)
    order = np.argsort(probs)[::-1][:8]
    y = 58
    for rank, idx in enumerate(order):
        prob = float(probs[idx])
        bar_w = int(prob * (pw - 80))
        col = C_ACCENT_2 if rank == 0 else C_ACCENT
        cv2.putText(image, ACTIONS[idx], (x0 + 16, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    C_TEXT if rank == 0 else C_TEXT_DIM, 2, cv2.LINE_AA)
        cv2.rectangle(image, (x0 + 46, y + 5), (x0 + 46 + (pw - 80), y + 22),
                      (50, 50, 55), -1)
        cv2.rectangle(image, (x0 + 46, y + 5), (x0 + 46 + bar_w, y + 22), col, -1)
        cv2.putText(image, f"{prob*100:4.0f}%", (x0 + pw - 28, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT_DIM, 1, cv2.LINE_AA)
        y += 34


def _draw_sentence_bar(image, sentence, top_conf, threshold):
    import cv2
    h, w = image.shape[:2]
    _rounded_panel(image, 0, 0, w - 230, 56, C_BG_PANEL, 0.78)
    text = ''.join(sentence) if sentence else "..."
    cv2.putText(image, text, (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_TEXT, 2, cv2.LINE_AA)
    meter_w = int((w - 230) * top_conf)
    cv2.rectangle(image, (0, 56), (w - 230, 60), (50, 50, 55), -1)
    col = C_ACCENT_2 if top_conf > threshold else C_ACCENT
    cv2.rectangle(image, (0, 56), (meter_w, 60), col, -1)


def _draw_status(image, fps, hand_present):
    import cv2
    h, w = image.shape[:2]
    _rounded_panel(image, 0, h - 34, 200, h, C_BG_PANEL, 0.7)
    cv2.putText(image, f"FPS {fps:4.1f}", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1, cv2.LINE_AA)
    dot = C_ACCENT_2 if hand_present else (80, 80, 90)
    cv2.circle(image, (130, h - 17), 7, dot, -1, cv2.LINE_AA)
    cv2.putText(image, "HAND", (145, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                C_TEXT if hand_present else C_TEXT_DIM, 1, cv2.LINE_AA)


def run(
    model_path: Optional[str] = None,
    camera: int = 0,
    threshold: float = DEFAULT_THRESHOLD,
    mirror: bool = True,
    window_name: str = "signspell — ASL Alphabet",
):
    """Launch the live recognizer with the pro UI.

    Keys:  q = quit | c = clear sentence | SPACE = add a space
    """
    import cv2

    rec = Recognizer(model_path=model_path, threshold=threshold)

    sentence, predictions = [], []
    probs = np.zeros(len(ACTIONS))
    prev_t = time.time()
    fps = 0.0

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        rec.close()
        raise RuntimeError(
            f"Could not open camera index {camera}. Try 0, 1, or 2."
        )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            letter, conf, p = rec.predict(frame)
            results = rec.last_results
            hand_present = results.right_hand_landmarks is not None
            if p is not None:
                probs = p

            # Draw on a BGR copy
            image = cv2.cvtColor(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR
            )
            _draw_hand(image, results)

            # Stabilise before committing a letter to the sentence
            if p is not None:
                predictions.append(int(np.argmax(probs)))
                if (
                    len(predictions) >= STABILITY_WINDOW
                    and len(np.unique(predictions[-STABILITY_WINDOW:])) == 1
                    and letter is not None
                ):
                    if not sentence or letter != sentence[-1]:
                        sentence.append(letter)
                if len(sentence) > 12:
                    sentence = sentence[-12:]

            now = time.time()
            inst = 1.0 / max(now - prev_t, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            prev_t = now

            top_conf = float(probs[np.argmax(probs)]) if probs.any() else 0.0
            _draw_prob_panel(image, probs)
            _draw_sentence_bar(image, sentence, top_conf, threshold)
            _draw_status(image, fps, hand_present)

            cv2.imshow(window_name, image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []
            elif key == ord(' '):
                sentence.append(' ')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rec.close()
