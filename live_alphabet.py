"""
Live ASL Alphabet Recognition
Matches the trained model: right-hand keypoints only (21 * 3 = 63),
30-frame sequences, 26 classes A-Z.

Run:  python live_alphabet.py
Quit: press 'q' in the video window.
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "alphabet_30_30_all_4.h5"   # change if your .h5 is elsewhere
CAMERA_INDEX = 0                          # 0 = default webcam (notebook used 1 for test)
SEQUENCE_LENGTH = 30
THRESHOLD = 0.5                           # min confidence to register a letter

actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                    'Y', 'Z'])

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return out, results


def draw_styled_landmarks(image, results):
    # Only the right hand was used for training
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def extract_keypoints(results):
    # Right hand only -> 21 landmarks * 3 (x, y, z) = 63 features
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z]
                       for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)
    return rh


def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        if 0.1 <= prob <= 1:
            cv2.rectangle(output_frame, (0, 100 + num * 35),
                          (int(prob * 100), 130 + num * 35), (16, 117, 245), -1)
            cv2.putText(output_frame, f"{actions[num]} {prob:.2f}",
                        (0, 125 + num * 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model loaded. Starting camera...")

    sequence = []
    sentence = []
    predictions = []

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {CAMERA_INDEX}. "
              f"Try changing CAMERA_INDEX (0, 1, 2...).")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predictions.append(np.argmax(res))

                # Stabilise: only accept if last 10 predictions agree
                if len(predictions) >= 10 and \
                        np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > THRESHOLD:
                        letter = actions[np.argmax(res)]
                        if len(sentence) > 0:
                            if letter != sentence[-1]:
                                sentence.append(letter)
                        else:
                            sentence.append(letter)

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image)

            # Top banner with the recognised sequence
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('ASL Alphabet - Live', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()