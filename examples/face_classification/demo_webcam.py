import cv2
import numpy as np
from tensorflow.keras.models import load_model
MODEL_PATH = 'experiments/FER_Baseline_20260202_191825/best_model.h5'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMAGE_SIZE = 48

print("loadingmodel")
model = load_model(MODEL_PATH)
print("model loaded")

print("initializing webcam")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("error: cannot access webcam")
    exit(1)


# Haar Cascade uses simple rectangular features to identify faces quickly

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    success, frame = camera.read()
    
    if not success:
        print("failed to capture")
        break
    grayfram = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = faceCascade.detectMultiScale(
        grayfram,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Loop through all detected faces
 
    for (x, y, w, h) in faces:
        faceregn = grayfram[y:y+h, x:x+w]
        faceresiz = cv2.resize(faceregn, (IMAGE_SIZE, IMAGE_SIZE))
        facenormalzd = faceresiz.astype('float32') / 255.0
        faceinput = facenormalzd.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
        predictions = model.predict(faceinput, verbose=0)
        emotionindx = np.argmax(predictions[0])
        emotionlabl = EMOTIONS[emotionindx]
        confidence = predictions[0][emotionindx] * 100
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotionlabl}: {confidence:.1f}%"

        # Draw label text above the face rectangle
        cv2.putText(frame, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    

    # Show frame with overlays in window
    cv2.imshow('emotion detection press Q to quit', frame)

    # If 'q' key is pressed (case-sensitive), break loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Important to free up camera for other applications
camera.release()
cv2.destroyAllWindows()
print("demo closed")







# code 2

"""
================================================================================
COSC-4427: Computer Vision Project
Real-Time Facial Emotion Detection - Webcam Demo

Author: Student 3 (Evaluation & Demo Lead)
Date: February 2026
Course: Computer Vision (Winter 2026)
Professor: Omar Al-Buraiki

Description:
This demo application performs real-time emotion detection using a webcam feed.
It detects faces in the video stream and classifies emotions using our trained
model. This demonstrates the practical application of our emotion recognition
system in a real-world scenario.

Features:
- Real-time face detection using Haar Cascade
- Live emotion prediction with confidence scores
- Visual overlay showing detected emotion
- 30 FPS performance for smooth user experience

Usage:
    python demo_webcam.py
    Press 'q' to quit

Requirements:
- Webcam/camera connected to computer
- Trained model file (best_model.h5)
- OpenCV library for video processing
================================================================================
"""

