# EmotiSense AI

A real-time facial and audio emotion detection web application built with Flask, DeepFace, and custom audio ML models.

## Features

- **Image analysis** — upload a photo to detect emotion, age, and gender
- **Live webcam** — capture and analyze faces in real time
- **Audio recording** — record speech and detect emotion from voice
- **History panel** — browse, select, and delete past analyses
- **Dark / light theme** — persistent across sessions

---

## Project structure

```
emotisense-ai/
├── app.py                  ← Flask backend (all routes)
├── Procfile                ← Render/Heroku start command
├── build.sh                ← Installs ffmpeg + Python packages
├── requirements.txt        ← Python dependencies
├── runtime.txt             ← Python version pin
├── .gitignore
│
├── templates/
│   └── index.html          ← Main UI (Jinja2 template)
│
├── static/
│   ├── css/
│   │   └── style.css       ← All styles (dark neural theme)
│   ├── js/
│   │   └── script.js       ← All frontend logic
│   └── uploads/            ← Temp storage for analyzed images
│
├── utils/
│   ├── __init__.py
│   ├── audio_emotion.py    ← Speech emotion prediction
│   └── audio_age_gender.py ← Age & gender from voice features
│
└── models/
    ├── README.md
    └── audio_emotion_model.h5   ← Your trained model (add manually)
```

---

## Local setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/emotisense-ai.git
cd emotisense-ai

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install ffmpeg (required for audio conversion)
# Ubuntu/Debian:
sudo apt-get install ffmpeg
# macOS:
brew install ffmpeg
# Windows: download from https://ffmpeg.org/download.html

# 5. Add your trained model
# Place audio_emotion_model.h5 inside the models/ folder

# 6. Run the app
python app.py
# Open http://localhost:5000
```

---

## Deploy to Render (free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set these values:
   - **Build command:** `bash build.sh`
   - **Start command:** `gunicorn app:app`
   - **Environment:** Python 3
5. Click Deploy — your app will be live at `https://your-app.onrender.com`

> **Note:** Render free tier sleeps after 15 min of inactivity. First request after sleep takes ~30s.

---

## API routes

| Method | Route | Description |
|---|---|---|
| GET | `/` | Serve the main UI |
| POST | `/predict_image` | Analyze uploaded image |
| POST | `/predict_audio` | Analyze live recorded audio |
| POST | `/predict_audio_file` | Analyze uploaded audio file |
| GET | `/get_history` | Return all history entries |
| POST | `/delete_history_selected` | Delete selected entries |
| GET | `/restore_history` | Restore from backup |

---

## Tech stack

- **Backend:** Python, Flask, Gunicorn
- **Face analysis:** DeepFace (emotion, age, gender)
- **Audio:** librosa, pydub, ffmpeg, custom Keras model
- **Frontend:** Vanilla JS, CSS custom properties
- **Deployment:** Render.com

---
title: EmotiSense AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# EmotiSense AI

A real-time facial and audio emotion detection web application built with Flask and DeepFace.

## Features
- Image emotion detection
- Live webcam analysis
- Audio emotion recognition
- Analysis history