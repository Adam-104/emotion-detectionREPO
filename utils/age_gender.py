import cv2
import numpy as np

ageNet = cv2.dnn.readNet(
    "models/age_net.caffemodel",
    "models/age_deploy.prototxt"
)

genderNet = cv2.dnn.readNet(
    "models/gender_net.caffemodel",
    "models/gender_deploy.prototxt"
)

AGE_LIST = ['0-12','13-19','20-35','36-55','56+']
GENDER_LIST = ['Male','Female']


def categorize(age):
    if age == '0-12':
        return "Child"
    elif age == '13-19':
        return "Teen"
    elif age == '20-35':
        return "Young Adult"
    elif age == '36-55':
        return "Adult"
    else:
        return "Senior"


def predict_age_gender(image_path):
    img = cv2.imread(image_path)

    blob = cv2.dnn.blobFromImage(
        img, 1.0, (227,227),
        (78.426,87.768,114.895),
        swapRB=False
    )

    # -------- Gender Prediction --------
    genderNet.setInput(blob)
    gender_preds = genderNet.forward()[0]

    gender_index = np.argmax(gender_preds)
    confidence = gender_preds[gender_index]

    gender = GENDER_LIST[gender_index]

    # 🔥 Confidence correction
    if confidence < 0.6:
        gender = "Unknown"

    # -------- Age Prediction --------
    ageNet.setInput(blob)
    age = AGE_LIST[ageNet.forward().argmax()]

    category = categorize(age)

    return category, gender