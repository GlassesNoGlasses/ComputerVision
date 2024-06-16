
import cv2
import numpy as np
import os
from PIL import Image

# Keyboard inputs
ESC = 27

# saving face image logic
MAX_FACES = 10
THRESHOLD = (500, 500)

## make faces directory to store faces if it doesn't exist
if not os.path.exists('./faces'):
    os.makedirs('faces')

# camera
camera = cv2.VideoCapture(1)

# face classifier from OpenCV
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def padRectangle(x, y, w, h, size=500):
    ''' Pad the rectangle around the face to be 500x500.'''

    width_padding = size - w
    height_padding = size - h

    return (x, y, w + width_padding, h + height_padding)

def getFaceCoords(gray_frame):
    ''' Outlines the face in the frame if it exists.
        Return the coordinates of rectangle if face exists.'''
    
    # return coordinates
    coordinates = None

    # get the faces
    faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

    # get rectangle around faces
    for (x, y, w, h) in faces:
        coordinates = padRectangle(x, y, w, h)
    
    return coordinates


def filterImageBySize(image, size=THRESHOLD):
    ''' Return True if the size of the image passes the threshold specifications.'''

    if (not image or not size):
        return False
    
    width, height = size

    return image.size[0] >= width and image.size[1] >= height


def getFaces():
    ''' Capture the faces from the camera feed and save them to the faces directory.'''

    # number of current faces stored
    num_faces = 0

    # loop through the camera feed
    while num_faces < MAX_FACES:
        # read the camera
        ret, frame = camera.read()
    
        if not ret:
            break
    
        # convert the frame to RGB instead of BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # get face coordinates
        coords = getFaceCoords(gray)

        # default face frame
        face_frame = frame
    
        # get the face coordinates
        if (coords and filterImageBySize(Image.fromarray(frame))):
            num_faces += 1
            x, y, w, h = coords
            face_frame = frame[y:y+h, x:x+w]
            cv2.imwrite(f'./faces/face_{num_faces}.jpg', face_frame)
        
        # stop the camera after 10 faces
        if (num_faces >= MAX_FACES):
            break
        
        # TODO: remove after testing
        if (coords):
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (255, 0, 0), 2)
        
        # display the frame
        cv2.imshow('Face Feed', frame)


# main function
def main():
    # get faces from the camera feed
    getFaces()
    print("Got all faces!")


main()
camera.release()
cv2.destroyAllWindows()
