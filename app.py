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
# FairFace returns age as ranges: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
# Trained on 108,501 balanced images across 7 races — works well for Indian/South Asian faces
FAIRFACE_AVAILABLE = False
fairface_predictor  = None

try:
    from uniface import RetinaFace as UFRetinaFace
    from uniface.attribute import FairFace
    print("Loading FairFace model...")
    fairface_predictor = FairFace(providers=["CPUExecutionProvider"])
    ff_detector        = UFRetinaFace(providers=["CPUExecutionProvider"])
    FAIRFACE_AVAILABLE = True
    print("FairFace loaded. Age/gender: FairFace mode.")
except Exception as e:
    print(f"FairFace not available ({e}), using InsightFace fallback.")
    try:
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        INSIGHTFACE_AVAILABLE = True
        print("InsightFace fallback loaded.")
    except Exception as e2:
        print(f"InsightFace also unavailable ({e2}), using DeepFace.")
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

# ─────────────── AGE RANGE → CATEGORY ───────────────
# FairFace age groups: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
AGE_CATEGORIES = {
    "0-2":   "Infant",
    "3-9":   "Child",
    "10-19": "Teenager",
    "20-29": "Young Adult",
    "30-39": "Adult",
    "40-49": "Middle Aged",
    "50-59": "Middle Aged",
    "60-69": "Senior",
    "70+":   "Elderly",
}

# ─────────────── SUGGESTION MAP ───────────────
# HSEmotion labels: happiness, sadness, anger, surprise, fear, disgust, neutral, contempt, excitement
# DeepFace labels:  happy, sad, angry, surprise, fear, disgust, neutral
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

# ─────────────── EMOTION DISPLAY NORMALIZATION ───────────────
EMOTION_DISPLAY = {
    "happiness":  "HAPPY",
    "sadness":    "SAD",
    "anger":      "ANGRY",
    "surprise":   "SURPRISE",
    "fear":       "FEAR",
    "disgust":    "DISGUST",
    "neutral":    "NEUTRAL",
    "contempt":   "CONTEMPT",
    "excitement": "EXCITEMENT",
    "happy":      "HAPPY",
    "sad":        "SAD",
    "angry":      "ANGRY",
}

def normalize_emotion(emotion):
    return EMOTION_DISPLAY.get(emotion.lower(), emotion.upper())

# ─────────────── IMAGE ENHANCEMENT ───────────────
def enhance_image(img_bgr):
    """CLAHE enhancement for dark/low-light images."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

# ─────────────── FACE DETECTION (OpenCV) ───────────────
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
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(img_bgr.shape[1], x + w + pad)
    y2  = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y1:y2, x1:x2]

# ─────────────── EMOTION — HSEmotion ───────────────
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

# ─────────────── AGE & GENDER — FairFace (primary) ───────────────
def get_age_gender_fairface(image_path):
    """
    FairFace via uniface:
    - Trained on 108K+ balanced images across 7 ethnicities
    - Returns age as range: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
    - Gender: Male / Female
    - Works well for South Asian, Indian, dark-skinned faces
    """
    try:
        img = cv2.imread(image_path)
        img = enhance_image(img)

        # Detect faces using RetinaFace
        faces = ff_detector.detect(img)
        if not faces:
            img_r = cv2.resize(img, (640, 640))
            faces = ff_detector.detect(img_r)

        if faces:
            # Pick largest face
            face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
            result = fairface_predictor.predict(img, face)

            age    = result.get("age", "Unknown")
            gender = result.get("gender", "Unknown")
            gender = gender.capitalize()

            print(f"FairFace result — age: {age}, gender: {gender}")
            return age, gender

    except Exception as e:
        print(f"FairFace error: {e}")

    return None, None

# ─────────────── AGE & GENDER — InsightFace (fallback 1) ───────────────
def get_age_gender_insightface(image_path):
    try:
        img   = enhance_image(cv2.imread(image_path))
        faces = face_app.get(img)
        if not faces:
            faces = face_app.get(cv2.resize(img, (640, 640)))
        if faces:
            face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]
            raw_age = face.age
            if raw_age is not None and not np.isnan(float(raw_age)):
                age = int(float(raw_age))
                # Age correction for low-light overestimation
                if age > 45:   age = age - 12
                elif age > 35: age = age - 8
                elif age > 25: age = age - 5
                elif age > 18: age = age - 3
                age = max(1, age)
                # Convert to range string for consistency
                if   age <= 2:  age_str = "0-2"
                elif age <= 9:  age_str = "3-9"
                elif age <= 19: age_str = "10-19"
                elif age <= 29: age_str = "20-29"
                elif age <= 39: age_str = "30-39"
                elif age <= 49: age_str = "40-49"
                elif age <= 59: age_str = "50-59"
                elif age <= 69: age_str = "60-69"
                else:           age_str = "70+"
            else:
                age_str = "Unknown"
            gender = "Male" if face.gender == 1 else "Female"
            return age_str, gender
    except Exception as e:
        print(f"InsightFace error: {e}")
    return None, None

# ─────────────── AGE & GENDER — DeepFace (fallback 2) ───────────────
def get_age_gender_deepface(image_path):
    try:
        result = DeepFace.analyze(
            image_path, actions=["age", "gender"],
            enforce_detection=False, detector_backend="opencv", silent=True
        )
        if isinstance(result, list): result = result[0]
        raw_age = int(result.get("age", 25))
        if   raw_age > 45: age = raw_age - 15
        elif raw_age > 35: age = raw_age - 10
        elif raw_age > 25: age = raw_age - 7
        else:              age = raw_age - 4
        age = max(1, age)
        if   age <= 2:  age_str = "0-2"
        elif age <= 9:  age_str = "3-9"
        elif age <= 19: age_str = "10-19"
        elif age <= 29: age_str = "20-29"
        elif age <= 39: age_str = "30-39"
        elif age <= 49: age_str = "40-49"
        elif age <= 59: age_str = "50-59"
        elif age <= 69: age_str = "60-69"
        else:           age_str = "70+"
        gender_raw = result.get("dominant_gender", "unknown").lower()
        gender = "Male" if gender_raw == "man" else "Female"
        return age_str, gender
    except Exception as e:
        print(f"DeepFace error: {e}")
        return "Unknown", "Unknown"

# ─────────────── MAIN AGE/GENDER DISPATCHER ───────────────
def get_age_gender(image_path):
    """Try FairFace → InsightFace → DeepFace in order."""
    if FAIRFACE_AVAILABLE:
        age, gender = get_age_gender_fairface(image_path)
        if age is not None:
            return age, gender

    if 'face_app' in globals() and face_app is not None:
        age, gender = get_age_gender_insightface(image_path)
        if age is not None:
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
        raw_emotion      = predict_audio_emotion(wav_path)
        emotion          = normalize_emotion(raw_emotion)
        a_gender, a_age  = audio_age_gender(wav_path)
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
        raw_emotion      = predict_audio_emotion(wav_path)
        emotion          = normalize_emotion(raw_emotion)
        a_gender, a_age  = audio_age_gender(wav_path)
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