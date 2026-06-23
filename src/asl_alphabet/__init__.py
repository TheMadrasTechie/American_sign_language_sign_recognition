"""
signspell — live ASL fingerspelling alphabet recognition.

Quick start
-----------
    import signspell

    # Run the live recognizer with the bundled model and pro UI:
    signspell.run()

    # Or drive it yourself, frame by frame:
    rec = signspell.Recognizer()
    letter, confidence, probabilities = rec.predict(frame_bgr)
"""

from .recognizer import Recognizer
from .app import run
from .config import ACTIONS, SEQUENCE_LENGTH

__version__ = "0.1.0"
__all__ = ["Recognizer", "run", "ACTIONS", "SEQUENCE_LENGTH", "__version__"]
