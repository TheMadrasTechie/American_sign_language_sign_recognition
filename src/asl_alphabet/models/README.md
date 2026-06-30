# Bundled model

Place your trained model here as:

    alphabet_model.h5

This is the file `signspell` loads by default. It must:
- accept input shape (1, 30, 63)  — 30 frames x 63 right-hand keypoints
- output 26 class scores (A-Z softmax)

If you rename it, update MODEL_FILENAME in asl_alphabet/config.py.
