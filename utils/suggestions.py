def suggest(emotion):
    suggestions = {
        "Happy": "Keep smiling 😊",
        "Sad": "Talk to someone ❤️",
        "Angry": "Take deep breaths 😌",
        "Surprise": "Enjoy the moment 😲",
        "Neutral": "Stay focused 📚",
        "Fear": "Stay calm 💪",
        "Disgust": "Take a break 🌿"
    }
    return suggestions.get(emotion, "Stay positive!")