from flask import Flask, render_template, request, jsonify
import os, json, uuid, time
from datetime import datetime
from deepface import DeepFace
from utils.audio_emotion import predict_audio_emotion
from utils.audio_age_gender import predict_age_gender
from pydub import AudioSegment

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE  = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────── EMOTION → SUGGESTION MAP ───────────────
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
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        data = json.load(f)
        return [data] if isinstance(data, dict) else data

def save_history(entry):
    data = load_history()
    data.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)

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
    file   = request.files["image"]
    source = request.form.get("source", "image")

    filename = str(int(time.time() * 1000)) + ".jpg"
    path     = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        result = DeepFace.analyze(path, actions=["emotion", "age", "gender"],
                                  enforce_detection=False)
        if isinstance(result, list):
            result = result[0]

        emotion    = result["dominant_emotion"]
        age        = int(result["age"])
        gender_raw = result["dominant_gender"].lower()
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
    try:
        file      = request.files["audio"]
        filename  = str(int(time.time() * 1000))
        webm_path = os.path.join(UPLOAD_FOLDER, filename + ".webm")
        wav_path  = os.path.join(UPLOAD_FOLDER, filename + ".wav")

        file.save(webm_path)
        convert_to_wav(webm_path, wav_path)

        # Emotion from trained model
        emotion = predict_audio_emotion(wav_path)

        # Age & gender from acoustic features
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
    """Accepts an uploaded audio file (mp3, wav, ogg, webm, m4a)."""
    try:
        file      = request.files["audioFile"]
        ext       = os.path.splitext(file.filename)[1].lower() or ".webm"
        filename  = str(int(time.time() * 1000))
        raw_path  = os.path.join(UPLOAD_FOLDER, filename + ext)
        wav_path  = os.path.join(UPLOAD_FOLDER, filename + ".wav")

        file.save(raw_path)

        # Convert to wav if needed
        if ext != ".wav":
            convert_to_wav(raw_path, wav_path)
        else:
            wav_path = raw_path

        # Emotion from trained model
        emotion = predict_audio_emotion(wav_path)

        # Age & gender from acoustic features
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
    data    = request.get_json()
    times   = data.get("times", [])
    history = load_history()

    with open("backup_history.json", "w") as f:
        json.dump(history, f, indent=4)

    new_history = [item for item in history if item["time"] not in times]
    with open("history.json", "w") as f:
        json.dump(new_history, f, indent=4)

    return jsonify({"status": "deleted"})

@app.route("/restore_history")
def restore_history():
    if not os.path.exists("backup_history.json"):
        return jsonify({"status": "no_backup"})

    with open("backup_history.json", "r") as f:
        data = json.load(f)
    with open("history.json", "w") as f:
        json.dump(data, f, indent=4)

    return jsonify({"status": "restored"})

# ─────────────── RUN ───────────────

if __name__ == "__main__":
    app.run(debug=True)