import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Trained Model
model = load_model("model_file_30epochs.h5")

# Load Haar Cascade Face Detector
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

# Start Webcam
video = cv2.VideoCapture(0)

while True:

    ret, frame = video.read()

    if not ret:
        break

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

        # Draw Rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Label Background
        cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)

        # Emotion Text
        cv2.putText(
            frame,
            labels_dict[label],
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow("Live Emotion Detection", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()