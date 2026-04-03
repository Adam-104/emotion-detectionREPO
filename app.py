import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request, jsonify
import json, uuid, time
from datetime import datetime
from deepface import DeepFace
from utils.audio_emotion import predict_audio_emotion
from utils.audio_age_gender import predict_age_gender
from pydub import AudioSegment
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

app = Flask(__name__)

# ─────────────── CONFIG ───────────────
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ─────────────── MONGODB SETUP ───────────────
MONGO_URI = os.environ.get("MONGO_URI", "")

db_client     = None
history_col   = None
backup_col    = None

def get_db():
    """Lazy connect to MongoDB — only when first needed."""
    global db_client, history_col, backup_col
    if db_client is None:
        try:
            db_client   = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db_client.admin.command("ping")   # test connection
            db          = db_client["emotisense"]
            history_col = db["history"]
            backup_col  = db["backup"]
            print("MongoDB connected.")
        except ConnectionFailure as e:
            print(f"MongoDB connection failed: {e}")
            db_client = None
    return history_col, backup_col

# ─────────────── SUGGESTION MAP ───────────────
SUGGESTIONS = {
    "happy":    "Keep spreading that positive energy! 😊",
    "sad":      "It's okay to feel low — take it one step at a time. 💙",
    "angry":    "Take a deep breath. Things will get better. 🧘",
    "surprise": "Life is full of surprises — embrace them! 🌟",
    "fear":     "You're braver than you think. Face it step by step. 💪",
    "disgust":  "Try to shift focus to something you enjoy. 🌿",
    "neutral":  "Stay positive and keep moving forward! ✨",
    "contempt": "Practice empathy — it can change perspectives. 🤝",
}

def get_suggestion(emotion):
    return SUGGESTIONS.get(emotion.lower(), "Keep going — every emotion is valid! 💫")

# ─────────────── HISTORY UTILS ───────────────

def load_history():
    col, _ = get_db()
    if col is None:
        return []
    try:
        # Return all entries, newest first, exclude MongoDB _id
        entries = list(col.find({}, {"_id": 0}).sort("time", -1))
        return entries
    except Exception as e:
        print(f"Load history error: {e}")
        return []

def save_history(entry):
    col, _ = get_db()
    if col is None:
        print("MongoDB unavailable — history not saved.")
        return
    try:
        col.insert_one({**entry})
    except Exception as e:
        print(f"Save history error: {e}")

def backup_history():
    """Copy all history into backup collection."""
    col, bak = get_db()
    if col is None:
        return False
    try:
        entries = list(col.find({}, {"_id": 0}))
        if entries:
            bak.delete_many({})           # clear old backup
            bak.insert_many(entries)      # write fresh backup
        return True
    except Exception as e:
        print(f"Backup error: {e}")
        return False

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
        result = DeepFace.analyze(
            path,
            actions=["emotion", "age", "gender"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True
        )
        if isinstance(result, list):
            result = result[0]

        emotion    = result["dominant_emotion"]
        age        = int(result.get("age", 0))
        gender_raw = result.get("dominant_gender", "unknown").lower()
        gender     = "Male" if gender_raw == "man" else "Female"
        suggestion = get_suggestion(emotion)

    except Exception as e:
        print("Image Error:", e)
        emotion, age, gender = "No Face", "N/A", "N/A"
        suggestion = "Could not detect a face. Try a clearer image."

    entry = {
        "id":         str(uuid.uuid4()),
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       source,
        "image":      "/" + path.replace("\\", "/"),
        "emotion":    emotion,
        "age":        age,
        "gender":     gender,
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

        emotion     = predict_audio_emotion(wav_path)
        gender, age = predict_age_gender(wav_path)

    except Exception as e:
        print("Audio Error:", e)
        emotion, age, gender = "Neutral", "Unknown", "Unknown"

    suggestion = get_suggestion(emotion)

    entry = {
        "id":         str(uuid.uuid4()),
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       "audio",
        "emotion":    emotion,
        "age":        age,
        "gender":     gender,
        "suggestion": suggestion,
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

        emotion     = predict_audio_emotion(wav_path)
        gender, age = predict_age_gender(wav_path)

    except Exception as e:
        print("Audio File Error:", e)
        emotion, age, gender = "Neutral", "Unknown", "Unknown"

    suggestion = get_suggestion(emotion)

    entry = {
        "id":         str(uuid.uuid4()),
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       "audio",
        "emotion":    emotion,
        "age":        age,
        "gender":     gender,
        "suggestion": suggestion,
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
    data  = request.get_json()
    times = data.get("times", [])

    # backup first
    backup_history()

    col, _ = get_db()
    if col is None:
        return jsonify({"status": "error"}), 500

    try:
        col.delete_many({"time": {"$in": times}})
        return jsonify({"status": "deleted"})
    except Exception as e:
        print(f"Delete error: {e}")
        return jsonify({"status": "error"}), 500

@app.route("/restore_history")
def restore_history():
    col, bak = get_db()
    if col is None:
        return jsonify({"status": "no_backup"})

    try:
        entries = list(bak.find({}, {"_id": 0}))
        if not entries:
            return jsonify({"status": "no_backup"})

        col.delete_many({})        # clear current history
        col.insert_many(entries)   # restore from backup
        return jsonify({"status": "restored"})
    except Exception as e:
        print(f"Restore error: {e}")
        return jsonify({"status": "error"})

# ─────────────── RUN ───────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)