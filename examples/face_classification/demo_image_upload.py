import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
MODEL_PATH = 'experiments/FER_Baseline_20260202_191825/best_model.h5'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMAGE_SIZE = 48


# MAIN DETECTION FUNCTION
def detectemotionfromimage(imagepath):
    model = load_model(MODEL_PATH)
    image = cv2.imread(imagepath)
    
    if image is None:
        print("cannot load")
        return
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize Haar Cascade face detector
    facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # # Detect all faces in the image
    faces = facecascade.detectMultiScale(grayimage, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("no face detected")
        return

     
    # Loop through all detected faces
    for i, (x, y, w, h) in enumerate(faces):
        facereg = grayimage[y:y+h, x:x+w]
        faceresiz = cv2.resize(facereg, (IMAGE_SIZE, IMAGE_SIZE))
        facenormaliz = faceresiz.astype('float32') / 255.0
        faceinput = facenormaliz.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
        predictions = model.predict(faceinput, verbose=0)
        emotionind = np.argmax(predictions[0])
        emotionlabl = EMOTIONS[emotionind]
        confidence = predictions[0][emotionind] * 100
        print(f"detected emotion: {emotionlabl}")
        print(f"confidence: {confidence:.2f}%")

        # Draw green rectangle around face

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotionlabl}: {confidence:.1f}%"

        # Draw label text above face
        cv2.putText(image, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('press key to close', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    outputpth = imagepath.replace('.', '_result.')
    cv2.imwrite(outputpth, image)
    print(f"{outputpth}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python demo_image_upload.py <image_path>")
        print("python demo_image_upload.py test_face.jpg")
    else:
        detectemotionfromimage(sys.argv[1])











# code 2
"""
================================================================================
COSC-4427: Computer Vision Project
Static Image Emotion Detection - Image Upload Demo

Author: Student 3 (Evaluation & Demo Lead)
Date: February 2026
Course: Computer Vision (Winter 2026)
Professor: Omar Al-Buraiki

Description:
This demo application performs emotion detection on static images uploaded
by the user. It detects all faces in the image, classifies emotions, and
saves an annotated version showing the results.

Use Cases:
- Testing model on specific facial expressions
- Analyzing multiple faces in group photos
- Creating annotated examples for presentations
- Backup demo if webcam is unavailable

Usage:
    python demo_image_upload.py path/to/image.jpg
    
Examples:
    python demo_image_upload.py test_face.jpg
    python demo_image_upload.py family_photo.png
    python demo_image_upload.py celebrity_smiling.jpeg

Output:
    - Displays image with emotion labels
    - Saves annotated image as: original_name_result.extension
    - Prints emotion and confidence to console
================================================================================
"""

