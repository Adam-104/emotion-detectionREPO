import librosa
import numpy as np

def predict_audio_emotion(audio_path):
    y, sr = librosa.load(audio_path, duration=3)

    energy = np.mean(np.abs(y))

    if energy > 0.05:
        return "Angry"
    elif energy > 0.02:
        return "Happy"
    else:
        return "Sad"
    
