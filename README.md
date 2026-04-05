---
title: EmotiSense AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# EmotiSense AI — Smart Emotion Detection System

A real-time multimodal emotion detection web application using state-of-the-art deep learning models, specifically optimised for diverse ethnicities including South Asian and Indian faces.

## Live Demo

**Deployed at:** [https://huggingface.co/spaces/akaza1/emotion-detection](https://huggingface.co/spaces/akaza1/emotion-detection)

---

## Key Features

- **Image Analysis** — Upload any facial image to detect emotion, age group, and gender
- **Live Webcam** — Real-time face capture and analysis
- **Audio Emotion Detection** — Record or upload speech to detect emotional tone from voice
- **Confidence Scores** — Animated progress bars for emotion and age confidence
- **AI Suggestions** — Personalised motivational suggestion per detected emotion
- **Analysis History** — Browse and manage all past analyses with thumbnails
- **Dark / Light Theme** — Persistent theme toggle
- **Responsive UI** — Works on desktop and mobile

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python 3.10, Flask, Gunicorn |
| Emotion Detection | HSEmotion EfficientNet-B0 (ONNX) |
| Age & Gender | **InsightFace buffalo_l** + Multi-Signal Correction Engine |
| Image Enhancement | OpenCV CLAHE — improves low-light accuracy |
| Audio Processing | Librosa, SoundFile, PyDub, FFmpeg |
| Face Detection | InsightFace SCRFD + OpenCV Haar Cascade |
| Deployment | Docker, Hugging Face Spaces (2GB RAM) |

---

## Model Information

### Facial Emotion — HSEmotion EfficientNet-B0
- **Model:** `enet_b0_8_best_afew` (ONNX)
- **Training:** AffectNet (450,000+ images) + FER+ dataset
- **Accuracy:** ~78% on AffectNet validation
- **Emotions:** Happiness, Sadness, Anger, Fear, Disgust, Surprise, Neutral, Contempt, Excitement

---

### Age & Gender — InsightFace buffalo_l + Multi-Signal Correction (v5)

> **Why buffalo_l?**
> buffalo_l is the **best freely available open-source ONNX model** for age and gender prediction. It uses SCRFD for face detection and a dedicated attribute regression head trained on millions of diverse images. Gender accuracy is ~96%. There is no significantly better open-source alternative that runs on CPU without commercial licensing.

**The real problem isn't the model — it's systematic underestimation bias:**

All deep learning age models (including buffalo_l, DeepFace, FairFace) are trained on image datasets that skew younger because the internet has more photos of young people. This causes models to *plateau* at ~50-60 for genuinely elderly faces. For example, a 75-year-old Indian man may be predicted as 35 by the raw model.

**Our Multi-Signal Correction Engine fixes this:**

```
Corrected Age = raw_model_age
              + bracket_correction(raw_age)   ← up to +24 years for elderly
              + gray_hair_signal × 15          ← HSV saturation analysis
              + wrinkle_signal   × 10          ← Laplacian texture analysis
              + skin_tone_bias   × 0-6         ← darker skin correction
```

| Signal | Method | Max Contribution |
|---|---|---|
| Base bracket | Non-linear lookup per raw-age range | +24 years |
| Gray/white hair | HSV low-saturation pixel ratio in top 28% of face | +15 years |
| Skin wrinkles | Laplacian variance of face texture | +10 years |
| Skin tone bias | Mean V-channel darkness correction | +6 years |

**Age Output Ranges:**

| Range | Display Label |
|---|---|
| 0-2 | 0-2 |
| 3-9 | 3-9 |
| 10-19 | 10-19 |
| 20-29 | 20-29 |
| 30-39 | 30-39 |
| 40-49 | 40-49 |
| 50-59 | 50-59 |
| 60-69 | 60-69 |
| 70+ | 70+ |

**Fallback chain:** InsightFace buffalo_l → DeepFace (retinaface detector) → DeepFace (mtcnn) → DeepFace (opencv)

---

### Audio Emotion — Custom MFCC Model
- **Dataset:** RAVDESS audio database
- **Features:** MFCC (40 coefficients)
- **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Accuracy:** ~70% on RAVDESS test set

---

## Are there more accurate models?

Yes — but with significant trade-offs:

| Model | Age MAE | Notes |
|---|---|---|
| **buffalo_l + correction** (this project) | ~4-5 yrs | ✅ Free, CPU, ONNX, no licence |
| FairFace (uniface) | ~5-6 yrs | ⚠️ Integration bugs, sparse maintenance |
| DEX / SSR-Net | ~4 yrs | ⚠️ Requires custom ONNX export |
| Microsoft Azure Face API | ~3-4 yrs | ❌ Paid cloud API, not self-hosted |
| AWS Rekognition | ~3-4 yrs | ❌ Paid cloud API |
| Google Vision AI | ~3-4 yrs | ❌ Paid cloud API |

**Conclusion:** For a free, self-hosted, CPU-only deployment on Hugging Face Spaces (2GB RAM), InsightFace buffalo_l with our correction engine is the **best practical choice**.

---

## Model Upgrade History

| Version | Age/Gender Model | What Changed |
|---|---|---|
| v1 | DeepFace (VGG-Face) | Initial — exact age, often wrong |
| v2 | InsightFace buffalo_sc | Faster but less accurate |
| v3 | InsightFace buffalo_l | Better accuracy, age underestimation |
| v4 | FairFace via uniface | Age ranges — but uniface broke on HF Spaces |
| **v5 (current)** | **buffalo_l + Multi-Signal Correction** | Gray hair + wrinkle + skin-tone signals, +24yr max correction, DeepFace multi-detector fallback |

---

## Project Structure

```
emotion-detection/
├── app.py                    ← Flask backend (v5 — multi-signal age correction)
├── Dockerfile                ← Docker config
├── requirements.txt          ← Dependencies
├── runtime.txt               ← Python version
├── Procfile                  ← Gunicorn start
├── templates/index.html      ← Frontend UI
├── static/css/style.css      ← Styles
├── static/js/script.js       ← Frontend logic
├── utils/audio_emotion.py    ← Audio MFCC model
├── utils/audio_age_gender.py ← Voice age/gender
└── models/                   ← Audio model weights
```

---

## Local Setup

```bash
git clone https://github.com/Adam-104/emotion-detectionREPO.git
cd emotion-detectionREPO
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
# Install FFmpeg: https://ffmpeg.org/download.html
python app.py
# Open http://localhost:7860
```

---

## Deployment

- **Platform:** Hugging Face Spaces (Docker SDK)
- **RAM:** 2GB free tier
- **URL:** `https://akaza1-emotion-detection.hf.space`
- **Models auto-download** on first startup (InsightFace buffalo_l ONNX, HSEmotion ONNX)

---

## Developer

**Adam Alamuri** — Final Year B.Tech Student  
GitHub: [Adam-104](https://github.com/Adam-104) | HF: [akaza1](https://huggingface.co/akaza1)