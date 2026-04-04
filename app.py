import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request, jsonify
import json, uuid, time, cv2
import numpy as np
from datetime import datetime
from deepface import DeepFace
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from utils.audio_emotion import predict_audio_emotion
from utils.audio_age_gender import predict_age_gender as audio_age_gender
from pydub import AudioSegment

# ─────────────── LOAD HSEMOTION ───────────────
print("Loading HSEmotion model...")
fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
print("HSEmotion loaded.")

# ─────────────── LOAD INSIGHTFACE ───────────────
INSIGHTFACE_AVAILABLE = False
face_app = None

try:
    from insightface.app import FaceAnalysis
    print("Loading InsightFace model...")
    face_app = FaceAnalysis(
        name="buffalo_l",                        # full model — best accuracy
        providers=["CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    INSIGHTFACE_AVAILABLE = True
    print("InsightFace loaded. Age/gender: InsightFace mode.")
except Exception as e:
    print(f"InsightFace not available ({e}), using DeepFace.")

# ─────────────── FLASK APP ───────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE  = "history.json"
BACKUP_FILE   = "backup_history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ─────────────── SUGGESTION MAP ───────────────
SUGGESTIONS = {
    "happy":      "Keep spreading that positive energy! 😊",
    "sad":        "It's okay to feel low — take it one step at a time. 💙",
    "angry":      "Take a deep breath. Things will get better. 🧘",
    "surprise":   "Life is full of surprises — embrace them! 🌟",
    "fear":       "You're braver than you think. Face it step by step. 💪",
    "disgust":    "Try to shift focus to something you enjoy. 🌿",
    "neutral":    "Stay positive and keep moving forward! ✨",
    "contempt":   "Practice empathy — it can change perspectives. 🤝",
    "excitement": "Channel that energy into something creative! 🔥",
}

def get_suggestion(emotion):
    return SUGGESTIONS.get(emotion.lower(), "Keep going — every emotion is valid! 💫")

# ─────────────── FACE DETECTION ───────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face_crop(img_bgr):
    """Returns largest face crop with padding, or full image as fallback."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    if len(faces) == 0:
        return img_bgr
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    pad = int(0.2 * min(w, h))
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(img_bgr.shape[1], x + w + pad)
    y2  = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y1:y2, x1:x2]

# ─────────────── EMOTION ───────────────

def predict_emotion(image_path):
    """HSEmotion EfficientNet-B0 — ~78% accuracy."""
    try:
        img_bgr   = cv2.imread(image_path)
        if img_bgr is None:
            return "neutral", 0.0
        face_crop = detect_face_crop(img_bgr)
        face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        emotion, scores = fer.predict_emotions(face_rgb, logits=False)
        confidence = round(float(max(scores)) * 100, 1)
        return emotion.lower(), confidence
    except Exception as e:
        print(f"Emotion error: {e}")
        return "neutral", 0.0

# ─────────────── AGE & GENDER ───────────────

def get_age_gender(image_path):
    """
    InsightFace buffalo_l (best) → falls back to DeepFace.
    Handles None values from InsightFace gracefully.
    """

    # ── InsightFace path ──
    if INSIGHTFACE_AVAILABLE and face_app is not None:
        try:
            img   = cv2.imread(image_path)
            faces = face_app.get(img)

            # Retry with upscaled image if no faces detected
            if not faces:
                img_r = cv2.resize(img, (640, 640))
                faces = face_app.get(img_r)

            if faces:
                # Pick largest face by area
                face = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True
                )[0]

                # Safe age extraction — buffalo_l always provides age
                raw_age = face.age
                if raw_age is not None and not np.isnan(float(raw_age)):
                    age = int(float(raw_age))
                else:
                    age = "N/A"

                # Safe gender extraction
                raw_gender = face.gender
                if raw_gender is not None:
                    gender = "Male" if int(raw_gender) == 1 else "Female"
                else:
                    gender = "N/A"

                print(f"InsightFace result — age: {age}, gender: {gender}")
                return age, gender

        except Exception as e:
            print(f"InsightFace error: {e}")

    # ── DeepFace fallback ──
    try:
        result = DeepFace.analyze(
            image_path,
            actions=["age", "gender"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True
        )
        if isinstance(result, list):
            result = result[0]

        raw_age = int(result.get("age", 25))
        # DeepFace overestimates — apply correction
        age     = max(1, raw_age - 8) if raw_age > 20 else max(1, raw_age - 3)

        gender_raw = result.get("dominant_gender", "unknown").lower()
        gender     = "Male" if gender_raw == "man" else "Female"

        print(f"DeepFace fallback — age: {age}, gender: {gender}")
        return age, gender

    except Exception as e:
        print(f"DeepFace error: {e}")
        return "N/A", "N/A"

# ─────────────── HISTORY UTILS ───────────────

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
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
        print("History save error:", e)

# ─────────────── AUDIO UTILS ───────────────

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")

# ─────────────── ROUTES ───────────────

@app.route("/")
def home():
    return render_template("index.html")

# ── IMAGE ──

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
        emotion, confidence = predict_emotion(path)
        age, gender         = get_age_gender(path)
        suggestion          = get_suggestion(emotion)
    except Exception as e:
        print("Predict error:", e)
        emotion, age, gender = "No Face", "N/A", "N/A"
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

# ── AUDIO (live recording) ──

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
        emotion         = predict_audio_emotion(wav_path)
        a_gender, a_age = audio_age_gender(wav_path)
    except Exception as e:
        print("Audio Error:", e)
        emotion, a_age, a_gender = "Neutral", "Unknown", "Unknown"

    entry = {
        "id":         str(uuid.uuid4()),
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       "audio",
        "emotion":    emotion,
        "age":        a_age,
        "gender":     a_gender,
        "suggestion": get_suggestion(emotion),
        "image":      "",
    }
    save_history(entry)
    return jsonify(entry)

# ── AUDIO FILE UPLOAD ──

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
        emotion         = predict_audio_emotion(wav_path)
        a_gender, a_age = audio_age_gender(wav_path)
    except Exception as e:
        print("Audio File Error:", e)
        emotion, a_age, a_gender = "Neutral", "Unknown", "Unknown"

    entry = {
        "id":         str(uuid.uuid4()),
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       "audio",
        "emotion":    emotion,
        "age":        a_age,
        "gender":     a_gender,
        "suggestion": get_suggestion(emotion),
        "image":      "",
    }
    save_history(entry)
    return jsonify(entry)

# ── HISTORY ──

@app.route("/get_history")
def get_history():
    return jsonify(load_history())

@app.route("/delete_history_selected", methods=["POST"])
def delete_history_selected():
    data    = request.get_json()
    times   = data.get("times", [])
    history = load_history()
    try:
        with open(BACKUP_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except IOError as e:
        print("Backup error:", e)
    new_history = [i for i in history if i["time"] not in times]
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(new_history, f, indent=4)
    except IOError as e:
        print("Delete error:", e)
    return jsonify({"status": "deleted"})

@app.route("/restore_history")
def restore_history():
    if not os.path.exists(BACKUP_FILE):
        return jsonify({"status": "no_backup"})
    try:
        with open(BACKUP_FILE, "r") as f:
            data = json.load(f)
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except (json.JSONDecodeError, IOError) as e:
        print("Restore error:", e)
        return jsonify({"status": "error"})
    return jsonify({"status": "restored"})

# ─────────────── RUN ───────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)