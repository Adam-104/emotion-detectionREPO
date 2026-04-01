from deepface import DeepFace

def predict_all(image_path):
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']
        age = int(result[0]['age'])
        gender = result[0]['dominant_gender']

        return emotion.capitalize(), age, gender.capitalize()

    except:
        return "No Face Detected", "N/A", "N/A"