import cv2
import numpy as np
from tensorflow.keras.models import load_model
import winsound  # Only works on Windows

# Load the saved model
model = load_model('mask_detector_model.h5')

# Labels
labels = ['Mask', 'No Mask']

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize and normalize the frame
    img = cv2.resize(frame, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict mask / no mask
    prediction = model.predict(img)
    class_index = np.argmax(prediction[0])
    label = labels[class_index]
    confidence = prediction[0][class_index]

    # Alert box on the frame
    alert_color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
    alert_text = f"{label} ({confidence * 100:.1f}%)"
    cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), alert_color, -1)
    cv2.putText(frame, alert_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Optional sound alert
    if label == 'No Mask' and confidence > 0.90:
        winsound.Beep(1000, 200)  # frequency, duration in ms

    # Show the webcam feed with annotation
    cv2.imshow('Mask Detection Live', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
