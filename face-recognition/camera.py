
# Note: this file is used for testing purposes
#       of the camera and tracking faces.

import cv2
import numpy as np

# keyboard inputs
ESC = 27

# camera
camera = cv2.VideoCapture(1)

# face classifier from OpenCV
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)



while camera.isOpened():

    # break the loop if 'ESC' is pressed
    if (cv2.waitKey(1) & 0xFF == ESC):
        break

    # read the camera
    ret, frame = camera.read()

    if not ret:
        break

    # convert the frame to RGB instead of BGR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get the faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # draw the rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # display the frame
    cv2.imshow('feed', frame)


camera.release()
cv2.destroyAllWindows()
