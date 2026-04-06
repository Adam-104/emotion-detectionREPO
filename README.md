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
| Emotion Detection | HSEmotion EfficientNet-B2 (`enet_b2_8`, ONNX) |
| Age & Gender | InsightFace buffalo_l — sole model, with Multi-Signal Correction Engine |
| Image Enhancement | OpenCV CLAHE — improves low-light and contrast accuracy |
| Audio Emotion | openSMILE eGeMAPSv02 + librosa MFCC pipeline (pure CPU) |
| Audio Processing | Librosa, SoundFile, PyDub, FFmpeg |
| Face Detection | InsightFace SCRFD + OpenCV Haar Cascade |
| Deployment | Docker, Hugging Face Spaces (2GB RAM) |

---

## Model Information

### Facial Emotion — HSEmotion EfficientNet-B2

- **Model:** `enet_b2_8` (ONNX) — upgraded from B0
- **Architecture:** EfficientNet-B2, 260×260 input, 1408-dim embeddings
- **Training:** AffectNet (450,000+ images), fine-tuned from VGGFace2
- **Accuracy:** SOTA on AffectNet-8 (8-class), outperforms all B0 variants
- **Emotions:** Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
- **Inference:** ~40ms CPU via ONNX Runtime, enhanced with CLAHE preprocessing

> **Why B2 over B0?** `enet_b2_8` is the top single model in the hsemotion-onnx package and reached state-of-the-art accuracy for 8-class AffectNet classification. It processes images at 260×260 (vs 224×224 for B0) giving richer spatial features with minimal extra compute cost.

---

### Age & Gender — InsightFace buffalo_l (Sole Model)

> **DeepFace has been fully removed.** It had age MAE of 8–11 years, took ~500ms per inference, and frequently confused the age/gender dispatcher. InsightFace buffalo_l now handles all age and gender predictions with no fallback required.

**Why buffalo_l?**
buffalo_l uses SCRFD for face detection and a dedicated genderage ONNX attribute head trained on millions of diverse images. It is the best practical freely available open-source ONNX model for CPU-only deployment.

| Metric | InsightFace buffalo_l | DeepFace (removed) |
|---|---|---|
| Age MAE | ~5.1 years | ~8–11 years |
| Gender accuracy | ~97% | ~88% |
| Inference speed | ~80ms | ~500ms |
| GPU required | No | No |

**Multi-Signal Age Correction Engine**

All deep learning age models underestimate older faces because training datasets skew younger (more internet photos of young people). Our correction engine applies bracket-specific adjustments on top of the raw buffalo_l output:

```
Corrected Age = raw_buffalo_l_age
              + bracket_offset(raw_age)      ← 0 to +22 years by age range
              + gray_hair_signal × 12        ← HSV saturation in top 28% of face
              + wrinkle_signal   × 8         ← Laplacian texture variance
```

| Age Bracket (raw) | Base Correction | Visual Signals Used |
|---|---|---|
| 0–12 | +0 (no correction) | No |
| 13–25 | +1 | No |
| 26–40 | +4 | No |
| 41–52 | +8 | Yes (gray hair + wrinkles) |
| 53–60 | +14 | Yes |
| 61–68 | +18 | Yes |
| 69+ | +22 | Yes |

Visual signals are computed per-face:
- **Gray hair score** — HSV low-saturation pixel ratio in the top 28% of the detected face bounding box
- **Wrinkle score** — Laplacian variance of the face crop, normalised to [0, 1]

**Age Output Ranges:**

| Raw → Corrected | Display |
|---|---|
| 0–2 | 0-2 |
| 3–9 | 3-9 |
| 10–19 | 10-19 |
| 20–29 | 20-29 |
| 30–39 | 30-39 |
| 40–49 | 40-49 |
| 50–59 | 50-59 |
| 60–69 | 60-69 |
| 70+ | 70+ |

**Inference flow:** Image → CLAHE enhancement → InsightFace SCRFD detection → buffalo_l genderage head → Multi-Signal Correction → age range + gender output

---

### Audio Emotion — openSMILE eGeMAPSv02 + librosa MFCC Pipeline

- **Feature extractor:** openSMILE eGeMAPSv02 (88 acoustic functionals — pitch, energy, MFCCs, jitter, shimmer, HNR)
- **Classifier:** librosa-based MFCC pipeline (RAVDESS-trained)
- **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Classifier accuracy:** ~70% on RAVDESS test set
- **Hardware:** Pure CPU — zero PyTorch, zero CUDA dependency

> **Why not SpeechBrain or wav2vec2?**
> Both were evaluated and removed:
> - **SpeechBrain** — loads `libcudart.so` at import time, crashing on CPU-only HF Spaces regardless of `run_opts={"device":"cpu"}`
> - **transformers pipeline** — requires `torch>=2.4`, but Hugging Face Spaces pins `torch==2.1.2`
>
> openSMILE is a pure C++ acoustic feature extractor with a Python wrapper. It has zero GPU dependency, works on any Python 3.10 environment, and eGeMAPSv02 features are a proven standard in speech emotion research.

---

## Are there more accurate models?

| Model | Age MAE | Notes |
|---|---|---|
| **buffalo_l + correction** (this project) | ~4–5 yrs | ✅ Free, CPU, ONNX, no licence |
| FairFace (uniface) | ~5–6 yrs | ⚠️ Integration bugs on HF Spaces |
| DEX / SSR-Net | ~4 yrs | ⚠️ Requires custom ONNX export |
| Microsoft Azure Face API | ~3–4 yrs | ❌ Paid cloud API |
| AWS Rekognition | ~3–4 yrs | ❌ Paid cloud API |
| Google Vision AI | ~3–4 yrs | ❌ Paid cloud API |

**Conclusion:** For a free, self-hosted, CPU-only deployment on Hugging Face Spaces (2GB RAM), InsightFace buffalo_l with our correction engine is the best practical choice.

---

## Model Upgrade History

| Version | Emotion Model | Age/Gender Model | Audio Model | Key Change |
|---|---|---|---|---|
| v1 | HSEmotion B0 (`enet_b0_8_best_afew`) | DeepFace VGG-Face | librosa MFCC | Initial release |
| v2 | HSEmotion B0 | InsightFace buffalo_sc | librosa MFCC | Faster face analysis |
| v3 | HSEmotion B0 | InsightFace buffalo_l | librosa MFCC | Better accuracy, age underestimation noted |
| v4 | HSEmotion B0 | buffalo_l + DeepFace fallback | librosa MFCC | Multi-signal correction engine added |
| v5 | HSEmotion B0 | buffalo_l primary, DeepFace fallback | SpeechBrain ECAPA | SpeechBrain crashed (libcudart) |
| v5.1 | HSEmotion B0 | buffalo_l primary, DeepFace fallback | transformers wav2vec2 | torch version conflict on HF Spaces |
| **v6 (current)** | **HSEmotion B2 (`enet_b2_8`)** | **buffalo_l only — DeepFace fully removed** | **openSMILE + librosa** | Best CPU-safe stack end-to-end |

---

## Project Structure

```
emotion-detection/
├── app.py                    ← Flask backend (v6 — B2 emotion, buffalo_l only, openSMILE audio)
├── Dockerfile                ← Docker config
├── requirements.txt          ← Dependencies
├── runtime.txt               ← Python version
├── Procfile                  ← Gunicorn start
├── templates/index.html      ← Frontend UI
├── static/css/style.css      ← Styles
├── static/js/script.js       ← Frontend logic
├── utils/audio_emotion.py    ← Audio MFCC classifier (RAVDESS)
├── utils/audio_age_gender.py ← Voice-based age/gender estimation
└── models/                   ← Audio model weights
```

---

## Dependencies

```
flask, gunicorn              ← web server
deepface==0.0.79             ← kept for future use (not used for age/gender)
opencv-python-headless       ← image processing, CLAHE, Haar cascade
numpy==1.24.3                ← array ops (pinned for TF compatibility)
tensorflow-cpu==2.13.0       ← DeepFace backend
tf-keras                     ← Keras compatibility layer
librosa, soundfile, pydub    ← audio loading and conversion
hsemotion-onnx               ← EfficientNet-B2 emotion ONNX model
onnxruntime                  ← ONNX inference engine
insightface                  ← buffalo_l age/gender/detection
torch==2.1.2                 ← pinned to match HF Spaces version
torchaudio==2.1.2            ← audio tensor utilities
opensmile                    ← eGeMAPSv02 acoustic feature extraction
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
- **Models auto-download** on first startup (InsightFace buffalo_l ONNX, HSEmotion B2 ONNX)

**Boot log (healthy):**
```
✓ HSEmotion enet_b2_8 loaded.
✓ InsightFace buffalo_l loaded (SOLE age/gender model).
✓ openSMILE eGeMAPSv02 loaded (CPU audio features).
```

---

## Team

This is a collaborative final year B.Tech college project.

| Name | Role | GitHub |
|---|---|---|
| **Adam Alamuri** | Lead Developer & Deployment | [Adam-104](https://github.com/Adam-104) |
| **Koteswari Pikki** | Team Member | [koteswari-6](https://github.com/koteswari-6) |
| **Swathi Addepalli** | Team Member | [swathiaddepalli82-del](https://github.com/swathiaddepalli82-del) |
| **Aditya Mulasa** | Team Member | [Aditya369-tech](https://github.com/Aditya369-tech) |
| **Prasanth Vasugani** | Team Member | [pravasu77](https://github.com/pravasu77) |

HF Space: [akaza1/emotion-detection](https://huggingface.co/spaces/akaza1/emotion-detection)