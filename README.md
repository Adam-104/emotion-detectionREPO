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

A real-time multimodal emotion detection web application using state-of-the-art deep learning models, specifically optimized for diverse ethnicities including South Asian and Indian faces.

## Live Demo

**Deployed at:** [https://huggingface.co/spaces/akaza1/emotion-detection](https://huggingface.co/spaces/akaza1/emotion-detection)

---

## Key Features

- **Image Analysis** — Upload any facial image to detect emotion, age group, and gender
- **Live Webcam** — Real-time face capture and analysis
- **Audio Emotion Detection** — Record speech to detect emotional tone from voice
- **Confidence Scores** — Animated progress bars for emotion and age confidence
- **AI Suggestions** — Personalized motivational suggestion per emotion
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
| Age & Gender | **FairFace via uniface** (ONNX Runtime) |
| Image Enhancement | OpenCV CLAHE — improves low-light accuracy |
| Audio Processing | Librosa, SoundFile, PyDub, FFmpeg |
| Face Detection | RetinaFace (uniface) + OpenCV Haar Cascade |
| Deployment | Docker, Hugging Face Spaces (2GB RAM) |

---

## Model Information

### Facial Emotion — HSEmotion EfficientNet-B0
- **Model:** `enet_b0_8_best_afew` (ONNX)
- **Training:** AffectNet (450,000+ images) + FER+ dataset
- **Accuracy:** ~78% on AffectNet validation
- **Emotions:** Happiness, Sadness, Anger, Fear, Disgust, Surprise, Neutral, Contempt, Excitement

### Age & Gender — FairFace (via uniface)
- **Paper:** "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age" (WACV 2021)
- **Training:** 108,501 images balanced across **7 ethnicities** — White, Black, **Indian**, East Asian, Southeast Asian, Middle Eastern, Latino
- **Gender Accuracy:** ~95% across all ethnicities
- **Age Output:** Ranges — `0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`
- **Key Advantage:** Never gives embarrassingly wrong exact age — always returns a sensible age range
- **Why better than InsightFace:** Explicitly trained on Indian faces as a dedicated category; more honest with age ranges
- **Fallback chain:** FairFace → InsightFace buffalo_l → DeepFace

### Age Category Labels
| FairFace Range | Display Label |
|---|---|
| 0-2 | Infant |
| 3-9 | Child |
| 10-19 | Teenager |
| 20-29 | Young Adult |
| 30-39 | Adult |
| 40-49 | Middle Aged |
| 50-59 | Middle Aged |
| 60-69 | Senior |
| 70+ | Elderly |

### Audio Emotion — Custom MFCC Model
- **Dataset:** RAVDESS audio database
- **Features:** MFCC (40 coefficients)
- **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Accuracy:** ~70% on RAVDESS test set

---

## Model Upgrade History

| Version | Age/Gender Model | What Changed |
|---|---|---|
| v1 | DeepFace | Initial — exact age, often wrong |
| v2 | InsightFace buffalo_sc | Faster but less accurate |
| v3 | InsightFace buffalo_l | Better accuracy, overestimates dark faces |
| v4 (current) | **FairFace** | Age ranges, Indian face optimized, 3-tier fallback |

---

## Project Structure

```
emotion-detection/
├── app.py                    ← Flask backend
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
venv\Scripts\activate
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
- **Models auto-download** on first startup (FairFace ONNX, HSEmotion ONNX)

---

## Developer

**Adam Alamuri** — Final Year B.Tech Student
GitHub: [Adam-104](https://github.com/Adam-104) | HF: [akaza1](https://huggingface.co/akaza1)