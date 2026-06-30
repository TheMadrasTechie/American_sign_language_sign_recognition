"""Shared configuration and constants."""

from importlib import resources
import numpy as np

# Model input contract — must match how the model was trained.
ACTIONS = np.array(
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
)
SEQUENCE_LENGTH = 30          # frames per prediction window
NUM_KEYPOINTS = 21 * 3        # right hand: 21 landmarks x (x, y, z)

DEFAULT_THRESHOLD = 0.5
STABILITY_WINDOW = 10         # consecutive agreeing frames before a letter sticks

# Bundled model filename (lives in asl_alphabet/models/)
MODEL_FILENAME = "alphabet_model.h5"


def default_model_path() -> str:
    """Return the filesystem path to the model bundled inside the package."""
    with resources.as_file(
        resources.files("asl_alphabet.models") / MODEL_FILENAME
    ) as p:
        return str(p)
