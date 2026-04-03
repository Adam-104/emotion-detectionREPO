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

A real-time multimodal emotion detection web application that analyzes facial expressions and speech to detect human emotions with high accuracy, using state-of-the-art deep learning models optimized for diverse ethnicities including South Asian faces.

## Live Demo

**Deployed at:** [https://huggingface.co/spaces/akaza1/emotion-detection](https://huggingface.co/spaces/akaza1/emotion-detection)

---

## Project Overview

EmotiSense AI is a final year project demonstrating the application of deep learning and computer vision in real-world human emotion analysis. The system accepts three types of input — static images, live webcam feed, and audio recordings — and outputs the detected emotion, estimated age, gender, confidence scores, and personalized AI suggestions.

The project uses a carefully selected combination of models to maximize accuracy across all ethnicities, age groups, and lighting conditions.

---

## Key Features

- **Image Analysis** — Upload any facial image to detect emotion, age, and gender instantly
- **Live Webcam** — Capture and analyze facial expressions in real time via browser webcam
- **Audio Emotion Detection** — Record speech and detect emotional tone from voice
- **Confidence Scores** — Emotion and age confidence displayed as animated progress bars
- **AI Suggestions** — Personalized motivational suggestion based on detected emotion
- **Analysis History** — Browse, review, and manage all past analyses with thumbnails
- **Dark / Light Theme** — Full theme support with persistent preference
- **Responsive UI** — Works on desktop and mobile browsers

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python 3.10, Flask |
| Emotion Detection | HSEmotion EfficientNet-B2 (ONNX) |
| Age & Gender | InsightFace buffalo_sc model |
| Audio Processing | Librosa, SoundFile, PyDub, FFmpeg |
| Face Detection | OpenCV Haar Cascade + InsightFace RetinaFace |
| Model Runtime | ONNX Runtime (no GPU required) |
| Deployment | Docker, Hugging Face Spaces (2GB RAM) |

---

## Model Information

### Facial Emotion — HSEmotion EfficientNet-B2
- **Model:** `enet_b2_8_best_afew` (EfficientNet-B2)
- **Training Data:** AffectNet (450,000+ images) + FER+ dataset
- **Accuracy:** ~82% on validation set
- **Emotions Detected:** Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral, Contempt, Excitement
- **Runtime:** ONNX — fast CPU inference, no GPU required
- **Why HSEmotion B2:** Best lightweight emotion model available. Outperforms DeepFace default (~65%) by 17% and HSEmotion B0 (~78%) by 4%

### Age & Gender — InsightFace buffalo_sc
- **Model:** InsightFace `buffalo_sc` (RetinaFace + ArcFace based)
- **Training Data:** 5 million+ diverse faces across all ethnicities
- **Age Error:** ±4 years (vs DeepFace ±8 years)
- **Gender Accuracy:** 97% (vs DeepFace 88%)
- **Key Advantage:** Accurately predicts age and gender for South Asian, East Asian, and African faces — addresses the bias problem in most western-trained models
- **Runtime:** ONNX — fast CPU inference

### Audio Emotion — Custom MFCC Model
- **Features:** MFCC (Mel-Frequency Cepstral Coefficients), 40 coefficients
- **Model:** Custom trained Keras CNN on RAVDESS dataset
- **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Accuracy:** ~70% on RAVDESS test set

---

## Accuracy Comparison

### Before vs After Model Upgrade

| Feature | Old Model | New Model | Improvement |
|---|---|---|---|
| Emotion | DeepFace (~65%) | HSEmotion B2 (~82%) | +17% |
| Age error | DeepFace (±8 yr) | InsightFace (±4 yr) | 2x better |
| Gender | DeepFace (88%) | InsightFace (97%) | +9% |
| South Asian faces | Poor | Excellent | Major improvement |
| Child age detection | Overestimates (30+) | Accurate (±4 yr) | Fixed |

---

## Project Structure

```
emotion-detection/
├── app.py                    ← Flask backend (all API routes)
├── Dockerfile                ← Docker config for HF Spaces
├── requirements.txt          ← Python dependencies
├── runtime.txt               ← Python version
├── Procfile                  ← Gunicorn start command
│
├── templates/
│   └── index.html            ← Frontend UI (dark neural theme)
│
├── static/
│   ├── css/style.css         ← Styles and animations
│   ├── js/script.js          ← Frontend logic and API calls
│   └── uploads/              ← Temporary image/audio storage
│
├── utils/
│   ├── audio_emotion.py      ← MFCC extraction + model predict
│   └── audio_age_gender.py   ← Pitch-based age & gender from voice
│
└── models/
    └── audio_emotion_model.h5 ← Trained audio emotion model
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Serve the main UI |
| POST | `/predict_image` | Analyze image → emotion, age, gender, confidence |
| POST | `/predict_audio` | Analyze live recorded audio |
| POST | `/predict_audio_file` | Analyze uploaded audio file |
| GET | `/get_history` | Return all analysis history |
| POST | `/delete_history_selected` | Delete selected history entries |
| GET | `/restore_history` | Restore deleted history from backup |

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Adam-104/emotion-detectionREPO.git
cd emotion-detectionREPO

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install FFmpeg (required for audio)
# Windows: download from https://ffmpeg.org/download.html
# Ubuntu:  sudo apt-get install ffmpeg
# Mac:     brew install ffmpeg

# 5. Run the application
python app.py

# 6. Open in browser
# http://localhost:7860
```

---

## Deployment

The application is containerized with Docker and deployed on **Hugging Face Spaces**.

| Setting | Value |
|---|---|
| Platform | Hugging Face Spaces |
| SDK | Docker |
| RAM | 2GB free tier |
| Port | 7860 |
| Python | 3.10.13 |
| Server | Gunicorn (1 worker, 2 threads) |
| URL | `https://akaza1-emotion-detection.hf.space` |

### Model Downloads at Runtime
- **HSEmotion B2** — auto-downloads ONNX weights (~35MB) on first startup
- **InsightFace buffalo_sc** — auto-downloads model pack (~85MB) on first startup
- Both models are cached after first download — subsequent starts are instant

---

## Future Improvements

- MongoDB Atlas integration for persistent history across sessions
- Real-time video stream emotion tracking (frame-by-frame)
- Multi-face detection and group emotion analysis
- Emotion trend analytics dashboard with graphs
- Push notifications for emotion-based alerts
- Mobile app version (React Native)

---

## Developer

**Adam Alamuri** — Final Year B.Tech Student
GitHub: [Adam-104](https://github.com/Adam-104)
Hugging Face: [akaza1](https://huggingface.co/akaza1)