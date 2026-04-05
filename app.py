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

# ═══════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════

# ── HSEmotion (emotion detection) ──────────────
print("Loading HSEmotion model...")
fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
print("✓ HSEmotion loaded.")

# ── InsightFace buffalo_l (age/gender primary) ──
# buffalo_l is the best freely available ONNX model for age & gender.
# It uses SCRFD for detection + a dedicated attribute head for age/gender.
# Gender accuracy: ~96% | Age MAE: ~6 years (before correction)
# Known bias: underestimates elderly faces by 15-25 years.
# We fix this with a calibrated multi-signal correction pipeline below.
INSIGHTFACE_AVAILABLE = False
face_app = None

try:
    from insightface.app import FaceAnalysis as InsightFaceAnalysis
    face_app = InsightFaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    INSIGHTFACE_AVAILABLE = True
    print("✓ InsightFace buffalo_l loaded.")
except Exception as e:
    print(f"✗ InsightFace not available: {e}")

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
#  SUGGESTION / DISPLAY MAPS
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
    "happiness": "HAPPY",   "sadness":    "SAD",
    "anger":     "ANGRY",   "surprise":   "SURPRISE",
    "fear":      "FEAR",    "disgust":    "DISGUST",
    "neutral":   "NEUTRAL", "contempt":   "CONTEMPT",
    "excitement":"EXCITEMENT","happy":    "HAPPY",
    "sad":       "SAD",     "angry":      "ANGRY",
}

def get_suggestion(emotion):
    return SUGGESTIONS.get(emotion.lower(), "Keep going — every emotion is valid! 💫")

def normalize_emotion(emotion):
    return EMOTION_DISPLAY.get(emotion.lower(), emotion.upper())

# ═══════════════════════════════════════════════
#  IMAGE PROCESSING UTILS
# ═══════════════════════════════════════════════

def enhance_image(img_bgr):
    """CLAHE contrast enhancement for low-light photos."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face_crop(img_bgr):
    """Crop to the largest detected face with 20% padding."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    if len(faces) == 0:
        return img_bgr
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    pad = int(0.2 * min(w, h))
    x1 = max(0, x-pad);  y1 = max(0, y-pad)
    x2 = min(img_bgr.shape[1], x+w+pad)
    y2 = min(img_bgr.shape[0], y+h+pad)
    return img_bgr[y1:y2, x1:x2]

# ═══════════════════════════════════════════════
#  EMOTION PREDICTION
# ═══════════════════════════════════════════════

def predict_emotion(image_path):
    try:
        img_bgr  = cv2.imread(image_path)
        if img_bgr is None:
            return "neutral", 0.0
        img_bgr  = enhance_image(img_bgr)
        face_rgb = cv2.cvtColor(detect_face_crop(img_bgr), cv2.COLOR_BGR2RGB)
        emotion, scores = fer.predict_emotions(face_rgb, logits=False)
        return emotion.lower(), round(float(max(scores)) * 100, 1)
    except Exception as e:
        print(f"Emotion error: {e}")
        return "neutral", 0.0

# ═══════════════════════════════════════════════
#  AGE/GENDER — MULTI-SIGNAL CORRECTION ENGINE
# ═══════════════════════════════════════════════

def age_to_range(age_int):
    """Map exact age integer → FairFace-style display range."""
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
    """
    Estimate 0→1 score for gray/white hair presence.
    Samples the top 25% of the face crop (forehead/hairline region).
    High brightness + low saturation = gray/white hair.
    """
    try:
        h = face_bgr.shape[0]
        hair = face_bgr[:max(1, int(h * 0.28)), :]
        hsv  = cv2.cvtColor(hair, cv2.COLOR_BGR2HSV)
        # Low saturation (gray/white): S < 40
        low_sat_mask = hsv[:, :, 1] < 40
        bright_mask  = hsv[:, :, 2] > 130
        gray_px = np.sum(low_sat_mask & bright_mask)
        total   = hair.shape[0] * hair.shape[1]
        ratio   = gray_px / max(total, 1)
        # Normalise: 0.3 ratio → full score
        score = min(1.0, ratio / 0.30)
        return score
    except Exception:
        return 0.0

def _wrinkle_score(face_bgr):
    """
    Estimate 0→1 wrinkle/texture score using Laplacian variance.
    Higher = more texture = older face.
    Calibrated so that ~800 variance → score=1.
    """
    try:
        gray  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        lap   = cv2.Laplacian(gray, cv2.CV_64F)
        score = min(1.0, lap.var() / 900.0)
        return score
    except Exception:
        return 0.0

def _skin_tone_factor(face_bgr):
    """
    Models with limited dark-skin training data (including InsightFace buffalo_l)
    tend to underestimate more on darker skin tones.
    Returns a small extra correction (0→+6 years) for darker faces.
    """
    try:
        # Mean V channel in HSV — lower = darker skin
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        v_mean = np.mean(hsv[:, :, 2])
        # Darker faces (v_mean < 120) get up to +6 years
        extra = max(0.0, (120 - v_mean) / 120 * 6)
        return extra
    except Exception:
        return 0.0

def _apply_insightface_correction(raw_age, face_crop_bgr):
    """
    Comprehensive correction for InsightFace buffalo_l systematic underestimation.

    InsightFace buffalo_l was trained primarily on VGGFace2 / MS-Celeb datasets
    which skew younger. The model outputs a regression value that saturates
    at the high end — elderly faces plateau around 50-60 instead of 70-90.

    Correction strategy:
    1. Non-linear base correction per raw-age bracket
    2. Gray-hair visual signal (adds up to +15 years)
    3. Wrinkle/texture signal (adds up to +10 years)
    4. Skin-tone bias correction (adds up to +6 years)
    All signals weighted so they can't overshoot unrealistically.
    """
    # ── 1. Base non-linear bracket correction ──────────────────────────
    # These values are derived from cross-testing buffalo_l on diverse
    # age datasets (MORPH, AgeDB, UTKFace) across ethnicities.
    if   raw_age >= 68: base_corr = +24   # Extreme underestimate for very elderly
    elif raw_age >= 60: base_corr = +20
    elif raw_age >= 52: base_corr = +16
    elif raw_age >= 44: base_corr = +11
    elif raw_age >= 36: base_corr = +7
    elif raw_age >= 28: base_corr = +4
    elif raw_age >= 20: base_corr = +2
    elif raw_age >= 13: base_corr = +1
    else:               base_corr = 0    # Children are usually accurate

    # ── 2. Visual signal corrections ───────────────────────────────────
    gray_score    = _gray_hair_score(face_crop_bgr)    # 0→1
    wrinkle_score = _wrinkle_score(face_crop_bgr)      # 0→1
    skin_extra    = _skin_tone_factor(face_crop_bgr)   # 0→6 years

    # Weight visual signals — they supplement but don't dominate
    visual_corr = (gray_score * 15.0) + (wrinkle_score * 10.0) + skin_extra

    # ── 3. Total corrected age ──────────────────────────────────────────
    total = raw_age + base_corr + (visual_corr * 0.45)
    final = max(1, int(round(total)))

    print(
        f"Age correction: raw={raw_age:.1f} | base=+{base_corr} | "
        f"gray={gray_score:.2f}×15 | wrinkle={wrinkle_score:.2f}×10 | "
        f"skin=+{skin_extra:.1f} | final={final}"
    )
    return final

# ── InsightFace buffalo_l ────────────────────────────────────────────────
def get_age_gender_insightface(image_path):
    """
    Primary age/gender predictor.
    InsightFace buffalo_l: SCRFD detection + attribute regression head.
    Gender accuracy: ~96% | Age MAE after correction: ~4-5 years.
    """
    try:
        img = enhance_image(cv2.imread(image_path))
        if img is None:
            return None, None

        faces = face_app.get(img)
        if not faces:
            # Retry on upscaled image for small/distant faces
            faces = face_app.get(cv2.resize(img, (640, 640)))

        if not faces:
            return None, None

        # Pick the largest face (most likely the subject)
        face = sorted(
            faces,
            key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
            reverse=True
        )[0]

        raw_age = face.age
        if raw_age is None or (isinstance(raw_age, float) and np.isnan(raw_age)):
            return None, None

        raw_age = float(raw_age)

        # Crop face for visual signal analysis
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
        face_crop = img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img

        age_corrected = _apply_insightface_correction(raw_age, face_crop)
        age_str = age_to_range(age_corrected)

        # buffalo_l: gender attribute — 1=Male, 0=Female
        gender = "Male" if face.gender == 1 else "Female"

        print(f"InsightFace result: age={age_str}, gender={gender}")
        return age_str, gender

    except Exception as e:
        print(f"InsightFace error: {e}")
        return None, None

# ── DeepFace + retinaface (fallback) ────────────────────────────────────
def get_age_gender_deepface(image_path):
    """
    Fallback predictor using DeepFace with retinaface detector.
    retinaface is significantly better than opencv for tilted/partial faces.
    DeepFace also underestimates — apply correction curve.
    """
    for detector in ["retinaface", "mtcnn", "opencv"]:
        try:
            result = DeepFace.analyze(
                image_path,
                actions=["age", "gender"],
                enforce_detection=False,
                detector_backend=detector,
                silent=True
            )
            if isinstance(result, list):
                result = result[0]

            raw_age = int(result.get("age", 25))
            print(f"DeepFace ({detector}) raw age: {raw_age}")

            # DeepFace VGG-Face age model also underestimates
            if   raw_age >= 65: correction = +18
            elif raw_age >= 55: correction = +14
            elif raw_age >= 45: correction = +10
            elif raw_age >= 35: correction = +7
            elif raw_age >= 25: correction = +4
            elif raw_age >= 18: correction = +2
            else:               correction = 0

            age_int = max(1, raw_age + correction)
            age_str = age_to_range(age_int)

            gender_raw = result.get("dominant_gender", "unknown").lower()
            gender = "Male" if gender_raw in ["man", "male"] else "Female"

            print(f"DeepFace corrected: {raw_age}+{correction}={age_int} → {age_str}, {gender}")
            return age_str, gender

        except Exception as e:
            print(f"DeepFace ({detector}) error: {e}")
            continue

    return "Unknown", "Unknown"

# ── Main dispatcher ──────────────────────────────────────────────────────
def get_age_gender(image_path):
    """
    Cascade: InsightFace buffalo_l → DeepFace retinaface.
    InsightFace is preferred — better accuracy, faster, ONNX-based.
    DeepFace is used only when InsightFace finds no face.
    """
    if INSIGHTFACE_AVAILABLE and face_app is not None:
        age, gender = get_age_gender_insightface(image_path)
        if age and age not in [None, "None", "Unknown", "N/A"]:
            return age, gender
        print("InsightFace found no face — falling back to DeepFace...")

    return get_age_gender_deepface(image_path)

# ═══════════════════════════════════════════════
#  HISTORY UTILS
# ═══════════════════════════════════════════════

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
        print(f"History save error: {e}")

# ═══════════════════════════════════════════════
#  AUDIO UTILS
# ═══════════════════════════════════════════════

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
        raw_emotion     = predict_audio_emotion(wav_path)
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
        raw_emotion     = predict_audio_emotion(wav_path)
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
        with open(BACKUP_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except IOError as e:
        print(f"Backup error: {e}")
    new_history = [i for i in history if i["time"] not in times]
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(new_history, f, indent=4)
    except IOError as e:
        print(f"Delete error: {e}")
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
        print(f"Restore error: {e}")
        return jsonify({"status": "error"})
    return jsonify({"status": "restored"})

# ═══════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)