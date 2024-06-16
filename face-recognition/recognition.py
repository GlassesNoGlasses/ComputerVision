
import cv2
import numpy as np
import os
from PIL import Image

# Keyboard inputs
ESC = 27

# saving face image logic
num_faces = 0
THRESHOLD = (100, 100)

## make faces directory to store faces if it doesn't exist
if not os.path.exists('./faces'):
    os.makedirs('faces')

# camera
camera = cv2.VideoCapture(1)

# face classifier from OpenCV
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def getFaceCoords(gray_frame, dest_image):
    ''' Outlines the face in the frame if it exists.
        Return the coordinates of rectangle if face exists.'''
    
    # return coordinates
    coordinates = None

    # get the faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # pad the rectangle around the face
    padding = 20

    # draw the rectangle around the face
    for (x, y, w, h) in faces:
        coordinates = (x + padding, y + padding, w + padding, h + padding)
    
    return coordinates

def filterImageBySize(image, size=THRESHOLD):
    ''' Return True if the size of the image passes the threshold specifications.'''

    if (not image or not size):
        return False
    
    width, height = size

    return image.size[0] >= width and image.size[1] >= height

while camera.isOpened():
        # get keyboard input
        key = cv2.waitKey(33)

        # break the loop if 'ESC' is pressed
        if (key == ESC):
            break
    
        # read the camera
        ret, frame = camera.read()
    
        if not ret:
            break
    
        # convert the frame to RGB instead of BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # get face coordinates
        coords = getFaceCoords(gray, frame)
    
        # get the face coordinates
        if (coords and filterImageBySize(Image.fromarray(frame))):
            num_faces += 1
            x, y, w, h = coords
            cv2.imwrite(f'./faces/face_{num_faces}.jpg', frame[y:y+h, x:x+w])
        
        # stop the camera after 10 faces
        if (num_faces == 10):
            break
        
        # TODO: remove after testing
        if (coords):
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (255, 0, 0), 2)
        
        # display the frame
        cv2.imshow('feed', frame)

camera.release()
cv2.destroyAllWindows()
