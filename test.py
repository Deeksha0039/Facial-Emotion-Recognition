import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load Trained Model
model = load_model("model_file_30epochs.h5")

# Load Face Detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Emotion Labels
labels_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

# Read Image
image_path = "Training_98972491.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3
)

for (x, y, w, h) in faces:

    face = gray[y:y+h, x:x+w]

    resized = cv2.resize(face, (48, 48))

    normalized = resized / 255.0

    reshaped = np.reshape(normalized, (1, 48, 48, 1))

    prediction = model.predict(reshaped, verbose=0)

    label = np.argmax(prediction)

    print("Predicted Emotion :", labels_dict[label])

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)

    cv2.putText(
        frame,
        labels_dict[label],
        (x + 5, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

cv2.imshow("Emotion Detection", frame)

cv2.waitKey(0)

cv2.destroyAllWindows()