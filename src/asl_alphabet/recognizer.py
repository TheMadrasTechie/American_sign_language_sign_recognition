"""
Headless recognition engine.

The Recognizer wraps MediaPipe Holistic + the trained LSTM and exposes a
simple frame-in / prediction-out API. It holds the rolling 30-frame buffer
internally so callers just feed frames.
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np

from .config import (
    ACTIONS,
    SEQUENCE_LENGTH,
    NUM_KEYPOINTS,
    DEFAULT_THRESHOLD,
    default_model_path,
)


class Recognizer:
    """Live ASL alphabet recognizer.

    Parameters
    ----------
    model_path : str, optional
        Path to a Keras .h5 model. Defaults to the bundled model.
    threshold : float
        Minimum softmax confidence for a prediction to count as a letter.
    min_detection_confidence, min_tracking_confidence : float
        Passed straight to MediaPipe Holistic.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        # Imported lazily so `import signspell` is cheap and so that a missing
        # heavy dep surfaces a clear message only when you actually run.
        import mediapipe as mp
        from tensorflow.keras.models import load_model

        self.threshold = threshold
        # Model trained on TF 2.19 / Keras 3. compile=False loads only the
        # inference graph (no optimizer state needed) and avoids Keras 3
        # warnings about missing training config.
        self._model = load_model(model_path or default_model_path(), compile=False)

        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._sequence = deque(maxlen=SEQUENCE_LENGTH)
        self._last_results = None

    # -- public API --------------------------------------------------------
    def process(self, frame_bgr):
        """Run MediaPipe on a BGR frame; returns the raw results object."""
        import cv2

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self._holistic.process(image_rgb)
        self._last_results = results
        return results

    def predict(
        self, frame_bgr
    ) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """Feed one frame, return (letter, confidence, full_prob_vector).

        `letter` is None until the 30-frame buffer fills or when confidence
        is below threshold. `probabilities` is the length-26 softmax vector
        (or None before the buffer is full).
        """
        results = self.process(frame_bgr)
        self._sequence.append(self._extract_keypoints(results))

        if len(self._sequence) < SEQUENCE_LENGTH:
            return None, 0.0, None

        probs = self._model.predict(
            np.expand_dims(np.array(self._sequence), axis=0), verbose=0
        )[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        letter = ACTIONS[idx] if conf >= self.threshold else None
        return letter, conf, probs

    @property
    def last_results(self):
        """MediaPipe results from the most recent process()/predict() call."""
        return self._last_results

    def close(self):
        self._holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _extract_keypoints(results) -> np.ndarray:
        if results.right_hand_landmarks:
            return np.array(
                [[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]
            ).flatten()
        return np.zeros(NUM_KEYPOINTS)