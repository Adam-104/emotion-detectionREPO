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

A real-time multimodal emotion detection web application that analyzes facial expressions and speech to detect human emotions with high accuracy, optimized for diverse ethnicities including South Asian faces.

## Live Demo

**Deployed at:** [https://huggingface.co/spaces/akaza1/emotion-detection](https://huggingface.co/spaces/akaza1/emotion-detection)

---

## Project Overview

EmotiSense AI is a final year B.Tech project demonstrating the application of deep learning and computer vision in real-world human emotion analysis. The system accepts three types of input — static images, live webcam feed, and audio recordings — and outputs the detected emotion, estimated age, gender, confidence scores, and personalized AI suggestions.

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
| Backend | Python 3.10, Flask, Gunicorn |
| Emotion Detection | HSEmotion EfficientNet-B0 (ONNX) |
| Age & Gender | InsightFace buffalo_l (ONNX Runtime) |
| Image Enhancement | OpenCV CLAHE — improves low-light/dark selfie accuracy |
| Audio Processing | Librosa, SoundFile, PyDub, FFmpeg |
| Face Detection | OpenCV Haar Cascade |
| Deployment | Docker, Hugging Face Spaces (2GB RAM) |

---

## Model Information

### Facial Emotion — HSEmotion EfficientNet-B0
- **Model:** `enet_b0_8_best_afew` (ONNX format)
- **Training Data:** AffectNet (450,000+ images) + FER+ dataset
- **Accuracy:** ~78% on AffectNet validation set
- **Emotions Detected:** Happiness, Sadness, Anger, Fear, Disgust, Surprise, Neutral, Contempt, Excitement
- **Enhancement:** CLAHE image enhancement applied before inference for better accuracy on dark/low-light photos

### Age & Gender — InsightFace buffalo_l
- **Model:** Full InsightFace buffalo_l model pack
- **Training:** 5 million+ diverse faces across all ethnicities
- **Age Error:** ±4–6 years on well-lit photos
- **Gender Accuracy:** ~97%
- **Age Correction:** Adaptive correction applied based on estimated age range to compensate for overestimation in low-light/dark-skinned face photos
- **Fallback:** DeepFace with OpenCV backend if InsightFace unavailable

### Audio Emotion — Custom MFCC Model
- **Dataset:** RAVDESS (Ryerson Audio-Visual Database)
- **Features:** MFCC (40 coefficients, 174 frames)
- **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Accuracy:** ~70% on RAVDESS test set

---

## Accuracy Improvement History

| Version | Emotion Model | Age/Gender Model | Key Fix |
|---|---|---|---|
| v1 | DeepFace (~65%) | DeepFace (~88% gender) | Initial |
| v2 | HSEmotion B0 (~78%) | InsightFace buffalo_sc | +13% emotion accuracy |
| v3 | HSEmotion B0 (~78%) | InsightFace buffalo_l | Better age/gender |
| v4 (current) | HSEmotion B0 + CLAHE | buffalo_l + age correction | Fixed suggestion labels, better age for dark photos |

---

## Known Limitations

- Age prediction is less accurate for **dark/low-light selfies** — CLAHE enhancement helps but does not fully compensate
- Age prediction for **very dark skin tones in poor lighting** may still overestimate by 5–15 years
- Audio emotion requires clear speech — background noise reduces accuracy
- First startup takes ~45 seconds as InsightFace buffalo_l (~281MB) downloads automatically

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
│   └── index.html            ← Frontend UI
│
├── static/
│   ├── css/style.css         ← Dark neural theme styles
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

# 5. Run the application
python app.py

# 6. Open in browser
# http://localhost:7860
```

---

## Deployment

| Setting | Value |
|---|---|
| Platform | Hugging Face Spaces |
| SDK | Docker |
| RAM | 2GB free tier |
| Port | 7860 |
| Python | 3.10.13 |
| Server | Gunicorn (1 worker, 2 threads, 300s timeout) |
| URL | `https://akaza1-emotion-detection.hf.space` |

---

## Future Improvements

- MongoDB Atlas for persistent history across server restarts
- Real-time video stream emotion tracking (frame-by-frame)
- Multi-face detection for group analysis
- Emotion trend analytics dashboard with graphs
- Mobile application (React Native)
- Better low-light age estimation with dedicated nighttime face models

---

## Developer

**Adam Alamuri** — Final Year B.Tech Student
GitHub: [Adam-104](https://github.com/Adam-104)
Hugging Face: [akaza1](https://huggingface.co/akaza1)