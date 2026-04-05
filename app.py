import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

# ── Facial Emotion: EfficientNet-B2 (upgraded from B0) ──────────────────────
# enet_b2_8: EfficientNet-B2 trained on AffectNet-8, best single model in the package
# Valid model names: enet_b0_8_best_vgaf | enet_b0_8_best_afew | enet_b0_8_va_mtl | enet_b2_8 | enet_b2_7
print("Loading HSEmotion B2 model...")
fer = HSEmotionRecognizer(model_name='enet_b2_8')
print("✓ HSEmotion enet_b2_8 loaded.")

# ── Age/Gender PRIMARY: InsightFace buffalo_l ────────────────────────────────
# Flipped: InsightFace is now primary (80ms, MAE ~5 yrs), DeepFace is fallback
INSIGHTFACE_AVAILABLE = False
face_app = None
try:
    from insightface.app import FaceAnalysis as InsightFaceAnalysis
    face_app = InsightFaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    INSIGHTFACE_AVAILABLE = True
    print("✓ InsightFace buffalo_l loaded (PRIMARY for age/gender).")
except Exception as e:
    print(f"✗ InsightFace not available: {e}")

# ── Age/Gender FALLBACK: DeepFace ────────────────────────────────────────────
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✓ DeepFace loaded (FALLBACK for age/gender).")
except Exception as e:
    print(f"✗ DeepFace not available: {e}")

# ── Audio Emotion: SpeechBrain ECAPA-TDNN ───────────────────────────────────
# Force CPU before ANY torch/speechbrain import — prevents libcudart.so crash
# on CPU-only Hugging Face Spaces even when torch is installed with CUDA build
os.environ["SPEECHBRAIN_BACKEND"] = "cpu"
os.environ["TORCH_DEVICE"]        = "cpu"

SPEECHBRAIN_AVAILABLE = False
audio_emotion_classifier = None
try:
    import torch
    torch.set_num_threads(2)                  # keep CPU threads reasonable on HF Spaces

    from speechbrain.pretrained.interfaces import foreign_class
    audio_emotion_classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2Vec2Classifier",
        savedir="models/speechbrain_emotion",
        run_opts={"device": "cpu"}            # explicit CPU — prevents CUDA lookup
    )
    SPEECHBRAIN_AVAILABLE = True
    print("✓ SpeechBrain ECAPA emotion classifier loaded (CPU).")
except Exception as e:
    print(f"✗ SpeechBrain not available, falling back to librosa pipeline: {e}")

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
#  MAPS
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
    "happiness": "HAPPY",    "sadness":    "SAD",
    "anger":     "ANGRY",    "surprise":   "SURPRISE",
    "fear":      "FEAR",     "disgust":    "DISGUST",
    "neutral":   "NEUTRAL",  "contempt":   "CONTEMPT",
    "excitement":"EXCITEMENT","happy":     "HAPPY",
    "sad":       "SAD",      "angry":      "ANGRY",
}

# SpeechBrain IEMOCAP label map → our display labels
SPEECHBRAIN_LABEL_MAP = {
    "neu": "neutral",
    "hap": "happiness",
    "sad": "sadness",
    "ang": "anger",
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

def _get_face_crop_cv2(image_path):
    try:
        img  = enhance_image(cv2.imread(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48,48))
        if len(faces) == 0: return img
        x,y,w,h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        return img[max(0,y):min(img.shape[0],y+h), max(0,x):min(img.shape[1],x+w)]
    except Exception:
        return np.zeros((100,100,3), dtype=np.uint8)

# ═══════════════════════════════════════════════
#  EMOTION — EfficientNet-B2 (enet_b2_8_best_afew)
# ═══════════════════════════════════════════════
def predict_emotion(image_path):
    """
    Uses HSEmotionRecognizer with enet_b2_8_best_afew.
    Returns (emotion_str, confidence_float).
    """
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
#  AGE/GENDER — SMART CORRECTION ENGINE
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
    corrected = base + (visual_boost * 0.4)
    print(f"  Age correction: raw={raw_age:.1f} → base={base} + visual={visual_boost*0.4:.1f} = {int(round(corrected))}")
    return int(round(corrected))

# ── PRIMARY: InsightFace buffalo_l ───────────────────────────────────────────
def get_age_gender_insightface(image_path):
    """
    InsightFace buffalo_l: ~80ms inference, age MAE ~5.1 yrs.
    Now the PRIMARY dispatcher — much faster and more accurate than DeepFace.
    """
    if not INSIGHTFACE_AVAILABLE or face_app is None:
        return None, None
    try:
        img   = enhance_image(cv2.imread(image_path))
        faces = face_app.get(img)
        if not faces:
            faces = face_app.get(cv2.resize(img, (640, 640)))
        if not faces:
            return None, None

        face = sorted(
            faces,
            key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
            reverse=True
        )[0]

        raw_age = face.age
        if raw_age is None or np.isnan(float(raw_age)):
            return None, None

        raw_age   = float(raw_age)
        x1,y1,x2,y2 = [int(v) for v in face.bbox]
        face_crop = img[max(0,y1):min(img.shape[0],y2), max(0,x1):min(img.shape[1],x2)]

        corrected = _correct_age(raw_age, face_crop if face_crop.size > 0 else img)
        age_str   = age_to_range(max(0, corrected))
        gender    = "Male" if face.gender == 1 else "Female"

        print(f"InsightFace (primary): raw={raw_age:.1f} → {age_str}, {gender}")
        return age_str, gender

    except Exception as e:
        print(f"InsightFace error: {e}")
        return None, None

# ── FALLBACK: DeepFace ────────────────────────────────────────────────────────
def get_age_gender_deepface(image_path):
    """
    DeepFace: fallback only. Tries retinaface → mtcnn → opencv backends.
    """
    if not DEEPFACE_AVAILABLE:
        return None, None
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

            raw_age    = float(result.get("age", 25))
            gender_raw = result.get("dominant_gender", "Man").lower()
            gender     = "Male" if gender_raw in ["man", "male"] else "Female"

            print(f"DeepFace fallback ({detector}): raw_age={raw_age:.1f}, gender={gender}")

            face_crop = _get_face_crop_cv2(image_path)
            corrected = _correct_age(raw_age, face_crop)
            age_str   = age_to_range(max(0, corrected))

            print(f"  → Final: {age_str}, {gender}")
            return age_str, gender

        except Exception as e:
            print(f"DeepFace ({detector}) failed: {e}")
            continue
    return None, None

# ── Dispatcher (InsightFace first, DeepFace fallback) ─────────────────────────
def get_age_gender(image_path):
    # PRIMARY: InsightFace buffalo_l
    age, gender = get_age_gender_insightface(image_path)
    if age and age not in [None, "None", "Unknown"]:
        return age, gender

    # FALLBACK: DeepFace
    print("InsightFace failed — falling back to DeepFace...")
    age, gender = get_age_gender_deepface(image_path)
    if age and age not in [None, "None", "Unknown"]:
        return age, gender

    return "Unknown", "Unknown"

# ═══════════════════════════════════════════════
#  AUDIO EMOTION — SpeechBrain ECAPA-TDNN
# ═══════════════════════════════════════════════
def predict_audio_emotion_speechbrain(wav_path):
    """
    SpeechBrain ECAPA-TDNN fine-tuned on IEMOCAP.
    Returns normalised emotion string.
    Maps 4 IEMOCAP classes: neu/hap/sad/ang → our emotion labels.
    Falls back to librosa pipeline if SpeechBrain unavailable.
    """
    if SPEECHBRAIN_AVAILABLE and audio_emotion_classifier is not None:
        try:
            out_prob, score, index, label = audio_emotion_classifier.classify_file(wav_path)
            raw_label = label[0].strip().lower()
            emotion   = SPEECHBRAIN_LABEL_MAP.get(raw_label, raw_label)
            confidence = round(float(score.squeeze()) * 100, 1)
            print(f"SpeechBrain: {raw_label} → {emotion} ({confidence}%)")
            return emotion
        except Exception as e:
            print(f"SpeechBrain inference error: {e}")

    # Fallback to original librosa-based pipeline
    print("Falling back to librosa audio emotion pipeline...")
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
        raw_emotion     = predict_audio_emotion_speechbrain(wav_path)
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
        raw_emotion     = predict_audio_emotion_speechbrain(wav_path)
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