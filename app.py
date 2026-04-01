from flask import Flask, render_template, request, jsonify
import os, json, uuid, time
from datetime import datetime
from deepface import DeepFace
from utils.audio_emotion import predict_audio_emotion
from pydub import AudioSegment

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= HISTORY =================

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        data = json.load(f)
        if isinstance(data, dict):
            return [data]
        return data

def save_history(entry):
    data = load_history()
    data.append(entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ================= AUDIO UTILS =================

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")

# ================= IMAGE =================

@app.route("/predict_image", methods=["POST"])
def predict_image():

    file = request.files["image"]
    source = request.form.get("source", "image")

    filename = str(int(time.time()*1000)) + ".jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        result = DeepFace.analyze(path, actions=['emotion','age','gender'], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        emotion = result["dominant_emotion"]
        age = int(result["age"])

        gender_raw = result["dominant_gender"].lower()
        gender = "Male" if gender_raw == "man" else "Female"

        suggestion = "Stay positive!"

    except:
        emotion, age, gender = "No Face", "N/A", "N/A"
        suggestion = "Try again"

    entry = {
        "id": str(uuid.uuid4()),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": source,   # ✅ FIXED
        "image": "/" + path.replace("\\","/"),
        "emotion": emotion,
        "age": age,
        "gender": gender,
        "suggestion": suggestion
    }

    save_history(entry)

    return jsonify(entry)

# ================= AUDIO =================

@app.route("/predict_audio", methods=["POST"])
def predict_audio():

    try:
        file = request.files["audio"]

        filename = str(int(time.time()*1000))
        webm_path = os.path.join(UPLOAD_FOLDER, filename + ".webm")
        wav_path = os.path.join(UPLOAD_FOLDER, filename + ".wav")

        file.save(webm_path)
        convert_to_wav(webm_path, wav_path)

        emotion = predict_audio_emotion(wav_path)

    except Exception as e:
        print("Audio Error:", e)
        emotion = "Neutral"

    entry = {
        "id": str(uuid.uuid4()),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "audio",   # ✅ FIXED (VERY IMPORTANT)
        "emotion": emotion,
        "age": "N/A",
        "gender": "N/A",
        "suggestion": "Voice analysis",
        "image": ""
    }

    save_history(entry)

    return jsonify(entry)

# ================= HISTORY ROUTES =================

@app.route("/get_history")
def get_history():
    return jsonify(load_history())

@app.route("/delete_history_selected", methods=["POST"])
def delete_history_selected():

    data = request.get_json()
    times = data.get("times", [])

    history = load_history()

    # backup
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

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)