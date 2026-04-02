# Models

Place your trained model files here.

## Expected files

| File | Used by | Description |
|---|---|---|
| `audio_emotion_model.h5` | `utils/audio_emotion.py` | Trained Keras model for speech emotion recognition |

## Notes

- Model files are excluded from Git (see `.gitignore`) because they are large.
- Upload your model to Render using their **Disk** feature, or host it on Google Drive / Hugging Face and download it at startup.
- If you don't have a trained model yet, `utils/audio_emotion.py` will fall back to returning `"neutral"` so the app won't crash.

## Downloading model at startup (optional)

Add this to the top of `app.py` to auto-download your model from a URL on first boot:

```python
import urllib.request

MODEL_URL  = "https://your-storage-url/audio_emotion_model.h5"
MODEL_PATH = "models/audio_emotion_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")
```
