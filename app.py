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

# ─────────────── LOAD FAIRFACE via uniface ───────────────
# Using FaceAnalyzer with FairFace attribute — correct API
# face.sex  → 'Male' or 'Female'
# face.race → age range string e.g. '20-29'
# Note: uniface FairFace stores age in face.race (age group) and gender in face.sex
FAIRFACE_AVAILABLE = False
ff_analyzer = None

try:
    from uniface import FaceAnalyzer
    from uniface.attribute import FairFace
    print("Loading FairFace model via uniface FaceAnalyzer...")
    ff_analyzer = FaceAnalyzer(
        attributes=[FairFace(providers=["CPUExecutionProvider"])],
        providers=["CPUExecutionProvider"]
    )
    FAIRFACE_AVAILABLE = True
    print("FairFace loaded successfully.")
except Exception as e:
    print(f"FairFace not available: {e}")
    # Try InsightFace as fallback
    try:
        from insightface.app import FaceAnalysis as InsightFaceAnalysis
        face_app = InsightFaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        INSIGHTFACE_AVAILABLE = True
        print("InsightFace buffalo_l fallback loaded.")
    except Exception as e2:
        print(f"InsightFace also unavailable: {e2}")
        INSIGHTFACE_AVAILABLE = False
        face_app = None

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

def get_suggestion(emotion):
    return SUGGESTIONS.get(emotion.lower(), "Keep going — every emotion is valid! 💫")

EMOTION_DISPLAY = {
    "happiness":  "HAPPY",    "sadness":    "SAD",
    "anger":      "ANGRY",    "surprise":   "SURPRISE",
    "fear":       "FEAR",     "disgust":    "DISGUST",
    "neutral":    "NEUTRAL",  "contempt":   "CONTEMPT",
    "excitement": "EXCITEMENT","happy":     "HAPPY",
    "sad":        "SAD",      "angry":      "ANGRY",
}

def normalize_emotion(emotion):
    return EMOTION_DISPLAY.get(emotion.lower(), emotion.upper())

# ─────────────── IMAGE ENHANCEMENT ───────────────
def enhance_image(img_bgr):
    """CLAHE enhancement for dark/low-light photos."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

# ─────────────── FACE DETECTION ───────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face_crop(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    if len(faces) == 0:
        return img_bgr
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    pad = int(0.2 * min(w, h))
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(img_bgr.shape[1], x+w+pad), min(img_bgr.shape[0], y+h+pad)
    return img_bgr[y1:y2, x1:x2]

# ─────────────── EMOTION ───────────────
def predict_emotion(image_path):
    try:
        img_bgr   = cv2.imread(image_path)
        if img_bgr is None:
            return "neutral", 0.0
        img_bgr   = enhance_image(img_bgr)
        face_crop = detect_face_crop(img_bgr)
        face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        emotion, scores = fer.predict_emotions(face_rgb, logits=False)
        confidence = round(float(max(scores)) * 100, 1)
        return emotion.lower(), confidence
    except Exception as e:
        print(f"Emotion error: {e}")
        return "neutral", 0.0

# ─────────────── AGE & GENDER — FairFace ───────────────
def get_age_gender_fairface(image_path):
    """
    FairFace via uniface FaceAnalyzer.
    Returns age as range string and gender as Male/Female.
    face.sex  → gender string
    face.age  → age range string (0-2, 3-9, 10-19 ... 70+)
    """
    try:
        img = enhance_image(cv2.imread(image_path))
        faces = ff_analyzer.analyze(img)
        if faces:
            # Pick face with highest detection confidence
            face = sorted(faces, key=lambda f: f.confidence, reverse=True)[0]

            # FairFace age stored in face.age as range string
            age = getattr(face, 'age', None)
            # FairFace gender stored in face.sex
            sex = getattr(face, 'sex', None)

            if age is None:
                # Try alternate attribute names
                age = getattr(face, 'age_group', None) or getattr(face, 'race', None)
            if sex is None:
                sex = getattr(face, 'gender', None)

            # Normalize gender
            if sex:
                gender = "Female" if str(sex).lower() in ["female", "woman", "f"] else "Male"
            else:
                gender = "N/A"

            print(f"FairFace raw — age: {age}, sex: {sex} → gender: {gender}")
            return str(age) if age else None, gender

    except Exception as e:
        print(f"FairFace error: {e}")
    return None, None

# ─────────────── AGE & GENDER — InsightFace fallback ───────────────
def age_to_range(age_int):
    """Convert exact age integer to FairFace-style range string."""
    if   age_int <= 2:  return "0-2"
    elif age_int <= 9:  return "3-9"
    elif age_int <= 19: return "10-19"
    elif age_int <= 29: return "20-29"
    elif age_int <= 39: return "30-39"
    elif age_int <= 49: return "40-49"
    elif age_int <= 59: return "50-59"
    elif age_int <= 69: return "60-69"
    else:               return "70+"

def get_age_gender_insightface(image_path):
    try:
        img   = enhance_image(cv2.imread(image_path))
        faces = face_app.get(img)
        if not faces:
            faces = face_app.get(cv2.resize(img, (640, 640)))
        if faces:
            face = sorted(
                faces,
                key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                reverse=True
            )[0]
            raw_age = face.age
            if raw_age is not None and not np.isnan(float(raw_age)):
                age_int = int(float(raw_age))
                # Correction for overestimation
                if   age_int > 50: age_int -= 12
                elif age_int > 40: age_int -= 8
                elif age_int > 30: age_int -= 5
                elif age_int > 20: age_int -= 3
                age_int = max(1, age_int)
                age_str = age_to_range(age_int)
            else:
                age_str = "Unknown"
            gender = "Male" if face.gender == 1 else "Female"
            return age_str, gender
    except Exception as e:
        print(f"InsightFace error: {e}")
    return None, None

# ─────────────── AGE & GENDER — DeepFace fallback ───────────────
def get_age_gender_deepface(image_path):
    try:
        result = DeepFace.analyze(
            image_path, actions=["age", "gender"],
            enforce_detection=False, detector_backend="opencv", silent=True
        )
        if isinstance(result, list): result = result[0]
        raw_age = int(result.get("age", 25))
        if   raw_age > 50: age_int = raw_age - 15
        elif raw_age > 40: age_int = raw_age - 10
        elif raw_age > 30: age_int = raw_age - 7
        else:              age_int = raw_age - 4
        age_int = max(1, age_int)
        age_str = age_to_range(age_int)
        gender_raw = result.get("dominant_gender", "unknown").lower()
        gender = "Male" if gender_raw == "man" else "Female"
        return age_str, gender
    except Exception as e:
        print(f"DeepFace error: {e}")
        return "Unknown", "Unknown"

# ─────────────── MAIN DISPATCHER ───────────────
def get_age_gender(image_path):
    """FairFace → InsightFace → DeepFace cascade."""
    if FAIRFACE_AVAILABLE and ff_analyzer is not None:
        age, gender = get_age_gender_fairface(image_path)
        if age and age not in ["None", "Unknown"]:
            return age, gender

    if 'face_app' in dir() and face_app is not None:
        age, gender = get_age_gender_insightface(image_path)
        if age:
            return age, gender

    return get_age_gender_deepface(image_path)

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
        print("Predict error:", e)
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
        raw_emotion     = predict_audio_emotion(wav_path)
        emotion         = normalize_emotion(raw_emotion)
        a_gender, a_age = audio_age_gender(wav_path)
    except Exception as e:
        print("Audio Error:", e)
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
        raw_emotion     = predict_audio_emotion(wav_path)
        emotion         = normalize_emotion(raw_emotion)
        a_gender, a_age = audio_age_gender(wav_path)
    except Exception as e:
        print("Audio File Error:", e)
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