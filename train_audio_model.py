import os
import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

DATASET_PATH = "dataset"

X = []
y = []

# Extract MFCC
def extract_features(file_path):
    y_audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Emotion mapping
emotion_map = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise"
}

# Load dataset
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):

            emotion_code = file.split("-")[2]
            emotion = emotion_map[emotion_code]

            file_path = os.path.join(root, file)

            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(emotion)
            except:
                print("Error:", file_path)

X = np.array(X)
y = np.array(y)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

# Save model
pickle.dump(model, open("audio_model.pkl", "wb"))

print("Model trained and saved ✅")