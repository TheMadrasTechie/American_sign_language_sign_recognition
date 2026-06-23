# signspell

**Live ASL fingerspelling alphabet recognition — straight from your webcam.**

`signspell` recognises the American Sign Language manual alphabet (A–Z) in real
time using MediaPipe hand tracking and an LSTM model trained on 30-frame
keypoint sequences. It ships with a pretrained model and a polished webcam UI,
and it works both as a command-line tool and an importable library.

---

## Install

```bash
pip install signspell
```

> Requires Python 3.9–3.11. A webcam is required for live recognition.

## Run it (CLI)

```bash
signspell                  # default webcam, bundled model, pro UI
signspell --camera 1       # pick a different camera
signspell --threshold 0.6  # require higher confidence
signspell --no-mirror      # disable mirrored view
```

**In-window keys:** `q` quit · `c` clear sentence · `SPACE` add a space.

## Use it (library)

Run the full UI from code:

```python
import signspell
signspell.run()
```

Or drive recognition yourself, frame by frame:

```python
import cv2
import signspell

rec = signspell.Recognizer()
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    letter, confidence, probs = rec.predict(frame)
    if letter:
        print(letter, f"{confidence:.2f}")
```

The `Recognizer` keeps a rolling 30-frame buffer internally, so you just feed
frames and read predictions. `letter` is `None` until the buffer fills or when
confidence is below the threshold.

## How it works

1. **MediaPipe Holistic** extracts 21 right-hand landmarks per frame.
2. The last **30 frames** of `(x, y, z)` keypoints form a sequence.
3. An **LSTM** classifies the sequence into one of 26 letters.
4. A short stability window prevents flicker before a letter is committed.

## Bring your own model

```bash
signspell --model path/to/your_model.h5
```

```python
signspell.Recognizer(model_path="path/to/your_model.h5")
```

Your model must accept input shape `(1, 30, 63)` and output 26 class scores.

## License

MIT © Sundar
