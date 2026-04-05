# ── Set ALL env vars FIRST before any heavy imports ─────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"          # disable GPU globally
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TORCH_DEVICE"]          = "cpu"
os.environ["TRANSFORMERS_OFFLINE"]  = "0"

from flask import Flask, render_template, request, jsonify
import json, uuid, time, cv2
import numpy as np
from datetime import datetime
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from utils.audio_emotion import predict_audio_emotion
from utils.audio_age_gender import predict_age_gender as audio_age_gender
from pydub import AudioSegment

# ═══════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════

# ── 1. Facial Emotion: EfficientNet-B2 ──────────────────────────────────────
# enet_b2_8 = best available in hsemotion-onnx (AffectNet-8, SOTA accuracy)
# Valid names: enet_b0_8_best_vgaf | enet_b0_8_best_afew | enet_b0_8_va_mtl
#              enet_b2_8 | enet_b2_7
print("Loading HSEmotion enet_b2_8...")
fer = HSEmotionRecognizer(model_name='enet_b2_8')
print("✓ HSEmotion enet_b2_8 loaded.")

# ── 2. Age/Gender: InsightFace buffalo_l ONLY ────────────────────────────────
# DeepFace is REMOVED — it is slow (~500ms) and has MAE ~8-11 yrs vs
# InsightFace buffalo_l: ~80ms, MAE ~5.1 yrs, gender accuracy ~97%.
# No fallback needed — returns "Unknown" cleanly if no face detected.
INSIGHTFACE_AVAILABLE = False
face_app = None
try:
    from insightface.app import FaceAnalysis as InsightFaceAnalysis
    face_app = InsightFaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    INSIGHTFACE_AVAILABLE = True
    print("✓ InsightFace buffalo_l loaded (SOLE age/gender model).")
except Exception as e:
    print(f"✗ InsightFace not available: {e}")

# ── 3. Audio Emotion: wav2vec2 via HuggingFace transformers (CPU-safe) ────────
# SpeechBrain REMOVED — it loads libcudart.so at import time and crashes
# on CPU-only HF Spaces even with run_opts={"device":"cpu"}.
# Using transformers pipeline with device=-1 (pure CPU, no CUDA lookup).
# Model: superb/wav2vec2-base-superb-er (~360MB, 4-class, CPU-friendly)
AUDIO_CLASSIFIER_AVAILABLE = False
audio_pipe = None
try:
    import torch
    torch.set_num_threads(2)
    from transformers import pipeline as hf_pipeline
    audio_pipe = hf_pipeline(
        task="audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=-1                               # -1 = force CPU, no CUDA
    )
    AUDIO_CLASSIFIER_AVAILABLE = True
    print("✓ wav2vec2-base-superb-er audio classifier loaded (CPU).")
except Exception as e:
    print(f"✗ Audio classifier not available, using librosa fallback: {e}")

# ═══════════════════════════════════════════════
#  FLASK SETUP
# ═══════════════════════════════════════════════
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE  = "history.json"
BACKUP_FILE   = "backup_history.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ═══════════════════════════════════════════════
#  LABEL MAPS
# ═══════════════════════════════════════════════
SUGGESTIONS = {
    "happiness":  "Keep spreading that positive energy! 😊",
    "sadness":    "It's okay to feel low — take it one step at a time. 💙",
    "anger":      "Take a deep breath. Things will get better. 🧘",
    "surprise":   "Life is full of surprises — embrace them! 🌟",
    "fear":       "You're braver than you think. Face it step by step. 💪",
    "disgust":    "Try to shift focus to something you enjoy. 🌿",
    "neutral":    "Stay positive and keep moving forward! ✨",
    "contempt":   "Practice empathy — it can change perspectives. 🤝",
    "excitement": "Channel that energy into something creative! 🔥",
    "happy":      "Keep spreading that positive energy! 😊",
    "sad":        "It's okay to feel low — take it one step at a time. 💙",
    "angry":      "Take a deep breath. Things will get better. 🧘",
}
EMOTION_DISPLAY = {
    "happiness": "HAPPY",     "sadness":    "SAD",
    "anger":     "ANGRY",     "surprise":   "SURPRISE",
    "fear":      "FEAR",      "disgust":    "DISGUST",
    "neutral":   "NEUTRAL",   "contempt":   "CONTEMPT",
    "excitement":"EXCITEMENT", "happy":     "HAPPY",
    "sad":       "SAD",        "angry":     "ANGRY",
}

# superb/wav2vec2-base-superb-er label map → our emotion labels
SUPERB_LABEL_MAP = {
    "neu": "neutral",  "hap": "happiness",
    "sad": "sadness",  "ang": "anger",
    "neutral":   "neutral",   "happy":     "happiness",
    "sad":       "sadness",   "angry":     "anger",
    "happiness": "happiness", "sadness":   "sadness",
    "anger":     "anger",
}

def get_suggestion(emotion):
    return SUGGESTIONS.get(emotion.lower(), "Keep going — every emotion is valid! 💫")

def normalize_emotion(emotion):
    return EMOTION_DISPLAY.get(emotion.lower(), emotion.upper())

# ═══════════════════════════════════════════════
#  IMAGE UTILS
# ═══════════════════════════════════════════════
def enhance_image(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face_crop(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
    if len(faces) == 0:
        return img_bgr
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    pad = int(0.2 * min(w, h))
    return img_bgr[max(0,y-pad):min(img_bgr.shape[0],y+h+pad),
                   max(0,x-pad):min(img_bgr.shape[1],x+w+pad)]

# ═══════════════════════════════════════════════
#  FACIAL EMOTION — EfficientNet-B2
# ═══════════════════════════════════════════════
def predict_emotion(image_path):
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: return "neutral", 0.0
        img_bgr  = enhance_image(img_bgr)
        face_rgb = cv2.cvtColor(detect_face_crop(img_bgr), cv2.COLOR_BGR2RGB)
        emotion, scores = fer.predict_emotions(face_rgb, logits=False)
        confidence = round(float(max(scores)) * 100, 1)
        print(f"HSEmotion B2: {emotion} ({confidence}%)")
        return emotion.lower(), confidence
    except Exception as e:
        print(f"Emotion error: {e}")
        return "neutral", 0.0

# ═══════════════════════════════════════════════
#  AGE/GENDER — InsightFace buffalo_l ONLY
#  DeepFace fully removed (inaccurate MAE ~8-11 yrs, slow ~500ms)
# ═══════════════════════════════════════════════
def age_to_range(age_int):
    if   age_int <= 2:  return "0-2"
    elif age_int <= 9:  return "3-9"
    elif age_int <= 19: return "10-19"
    elif age_int <= 29: return "20-29"
    elif age_int <= 39: return "30-39"
    elif age_int <= 49: return "40-49"
    elif age_int <= 59: return "50-59"
    elif age_int <= 69: return "60-69"
    else:               return "70+"

def _gray_hair_score(face_bgr):
    try:
        h    = face_bgr.shape[0]
        hair = face_bgr[:max(1, int(h * 0.28)), :]
        hsv  = cv2.cvtColor(hair, cv2.COLOR_BGR2HSV)
        mask = (hsv[:,:,1] < 40) & (hsv[:,:,2] > 130)
        return min(1.0, np.sum(mask) / max(hair.shape[0]*hair.shape[1], 1) / 0.3)
    except Exception:
        return 0.0

def _wrinkle_score(face_bgr):
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 900.0)
    except Exception:
        return 0.0

def _correct_age(raw_age, face_crop_bgr):
    """
    Bracket-specific bias correction tuned for InsightFace buffalo_l.
    buffalo_l systematically underestimates older faces.
    Children (<=12): no correction — model is accurate here.
    """
    if raw_age <= 12:
        return int(round(raw_age))
    if raw_age <= 25:
        return int(round(raw_age + 1))
    if raw_age <= 40:
        return int(round(raw_age + 4))

    gray_score    = _gray_hair_score(face_crop_bgr)
    wrinkle_score = _wrinkle_score(face_crop_bgr)

    if   raw_age <= 52: base = raw_age + 8
    elif raw_age <= 60: base = raw_age + 14
    elif raw_age <= 68: base = raw_age + 18
    else:               base = raw_age + 22

    visual_boost = (gray_score * 12.0) + (wrinkle_score * 8.0)
    corrected    = base + (visual_boost * 0.4)
    print(f"  Age correction: raw={raw_age:.1f} base={base} visual={visual_boost*0.4:.1f} → {int(round(corrected))}")
    return int(round(corrected))

def get_age_gender(image_path):
    """
    InsightFace buffalo_l is the SOLE age/gender model.
    Gender: ~97% accuracy. Age MAE: ~5.1 yrs. Speed: ~80ms.
    """
    if not INSIGHTFACE_AVAILABLE or face_app is None:
        return "Unknown", "Unknown"
    try:
        img   = enhance_image(cv2.imread(image_path))
        faces = face_app.get(img)
        if not faces:
            faces = face_app.get(cv2.resize(img, (640, 640)))
        if not faces:
            print("InsightFace: no face detected.")
            return "Unknown", "Unknown"

        face = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )[0]

        raw_age = face.age
        if raw_age is None or np.isnan(float(raw_age)):
            return "Unknown", "Unknown"

        raw_age      = float(raw_age)
        x1,y1,x2,y2 = [int(v) for v in face.bbox]
        face_crop    = img[max(0,y1):min(img.shape[0],y2), max(0,x1):min(img.shape[1],x2)]

        corrected = _correct_age(raw_age, face_crop if face_crop.size > 0 else img)
        age_str   = age_to_range(max(0, corrected))
        gender    = "Male" if face.gender == 1 else "Female"

        print(f"InsightFace: raw={raw_age:.1f} → {corrected} → {age_str}, {gender}")
        return age_str, gender

    except Exception as e:
        print(f"InsightFace error: {e}")
        return "Unknown", "Unknown"

# ═══════════════════════════════════════════════
#  AUDIO EMOTION — wav2vec2 transformers pipeline
#  SpeechBrain removed (libcudart.so crash on CPU HF Spaces)
# ═══════════════════════════════════════════════
def predict_audio_emotion_model(wav_path):
    """
    Primary: superb/wav2vec2-base-superb-er (CPU-safe, device=-1).
    Fallback: original librosa pipeline from utils/audio_emotion.py.
    """
    if AUDIO_CLASSIFIER_AVAILABLE and audio_pipe is not None:
        try:
            import librosa
            speech, sr = librosa.load(wav_path, sr=16000, mono=True)
            results    = audio_pipe({"array": speech, "sampling_rate": sr})
            top_label  = results[0]["label"].lower().strip()
            emotion    = SUPERB_LABEL_MAP.get(top_label, top_label)
            score      = round(results[0]["score"] * 100, 1)
            print(f"wav2vec2 audio: {top_label} → {emotion} ({score}%)")
            return emotion
        except Exception as e:
            print(f"wav2vec2 inference error: {e}")

    print("Using librosa audio emotion fallback...")
    return predict_audio_emotion(wav_path)

# ═══════════════════════════════════════════════
#  HISTORY
# ═══════════════════════════════════════════════
def load_history():
    if not os.path.exists(HISTORY_FILE): return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            return [data] if isinstance(data, dict) else data
    except (json.JSONDecodeError, IOError):
        return []

def save_history(entry):
    data = load_history()
    data.append(entry)
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"History save error: {e}")

def convert_to_wav(input_path, output_path):
    AudioSegment.from_file(input_path).export(output_path, format="wav")

# ═══════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    source   = request.form.get("source", "image")
    filename = str(int(time.time() * 1000)) + ".jpg"
    path     = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        raw_emotion, confidence = predict_emotion(path)
        emotion                 = normalize_emotion(raw_emotion)
        age, gender             = get_age_gender(path)
        suggestion              = get_suggestion(raw_emotion)
    except Exception as e:
        print(f"Predict error: {e}")
        emotion, age, gender = "NO FACE", "N/A", "N/A"
        confidence = 0.0
        suggestion = "Could not detect a face. Try a clearer image."

    entry = {
        "id":         str(uuid.uuid4()),
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       source,
        "image":      "/" + path.replace("\\", "/"),
        "emotion":    emotion,
        "age":        age,
        "gender":     gender,
        "confidence": confidence,
        "suggestion": suggestion,
    }
    save_history(entry)
    return jsonify(entry)

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    file = request.files.get("audio")
    if not file:
        return jsonify({"error": "No audio provided"}), 400
    try:
        filename  = str(int(time.time() * 1000))
        webm_path = os.path.join(UPLOAD_FOLDER, filename + ".webm")
        wav_path  = os.path.join(UPLOAD_FOLDER, filename + ".wav")
        file.save(webm_path)
        convert_to_wav(webm_path, wav_path)
        raw_emotion     = predict_audio_emotion_model(wav_path)
        emotion         = normalize_emotion(raw_emotion)
        a_gender, a_age = audio_age_gender(wav_path)
    except Exception as e:
        print(f"Audio error: {e}")
        emotion, a_age, a_gender, raw_emotion = "NEUTRAL", "Unknown", "Unknown", "neutral"

    entry = {
        "id":         str(uuid.uuid4()),
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       "audio",
        "emotion":    emotion,
        "age":        a_age,
        "gender":     a_gender,
        "suggestion": get_suggestion(raw_emotion),
        "image":      "",
    }
    save_history(entry)
    return jsonify(entry)

@app.route("/predict_audio_file", methods=["POST"])
def predict_audio_file():
    file = request.files.get("audioFile")
    if not file:
        return jsonify({"error": "No audio file provided"}), 400
    try:
        ext      = os.path.splitext(file.filename)[1].lower() or ".webm"
        filename = str(int(time.time() * 1000))
        raw_path = os.path.join(UPLOAD_FOLDER, filename + ext)
        wav_path = os.path.join(UPLOAD_FOLDER, filename + ".wav")
        file.save(raw_path)
        if ext != ".wav":
            convert_to_wav(raw_path, wav_path)
        else:
            wav_path = raw_path
        raw_emotion     = predict_audio_emotion_model(wav_path)
        emotion         = normalize_emotion(raw_emotion)
        a_gender, a_age = audio_age_gender(wav_path)
    except Exception as e:
        print(f"Audio file error: {e}")
        emotion, a_age, a_gender, raw_emotion = "NEUTRAL", "Unknown", "Unknown", "neutral"

    entry = {
        "id":         str(uuid.uuid4()),
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       "audio",
        "emotion":    emotion,
        "age":        a_age,
        "gender":     a_gender,
        "suggestion": get_suggestion(raw_emotion),
        "image":      "",
    }
    save_history(entry)
    return jsonify(entry)

@app.route("/get_history")
def get_history():
    return jsonify(load_history())

@app.route("/delete_history_selected", methods=["POST"])
def delete_history_selected():
    data    = request.get_json()
    times   = data.get("times", [])
    history = load_history()
    try:
        with open(BACKUP_FILE, "w") as f: json.dump(history, f, indent=4)
    except IOError as e:
        print(f"Backup error: {e}")
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump([i for i in history if i["time"] not in times], f, indent=4)
    except IOError as e:
        print(f"Delete error: {e}")
    return jsonify({"status": "deleted"})

@app.route("/restore_history")
def restore_history():
    if not os.path.exists(BACKUP_FILE):
        return jsonify({"status": "no_backup"})
    try:
        with open(BACKUP_FILE, "r") as f: data = json.load(f)
        with open(HISTORY_FILE, "w") as f: json.dump(data, f, indent=4)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Restore error: {e}")
        return jsonify({"status": "error"})
    return jsonify({"status": "restored"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)