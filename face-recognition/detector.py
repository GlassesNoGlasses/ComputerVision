
import cv2
import numpy as np
import os
import csv
from PIL import Image

# saving face image logic
MAX_FACES = 25
IMG_SIZES = (500, 500)
USER_FACES_DIR = './faces'
TEST_FACES_DIR = './test_faces'


class FaceDetector:
    ''' Class to detect faces in the camera feed.'''

    def __init__(self, user_faces_dir=USER_FACES_DIR, test_faces_dir=TEST_FACES_DIR, max_faces=MAX_FACES):
        ''' Initialize the camera, face classifier, and directories.'''

        # camera
        self.camera = cv2.VideoCapture(1)

        # face classifier from OpenCV
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.user_faces_dir = user_faces_dir
        self.test_faces_dir = test_faces_dir
        self.max_faces = max_faces
        self.img_sizes = IMG_SIZES

    def initializeFiles(self):
        ''' Initialize the files/dirs for the face recognition system.'''

        print("Initializing files...")

        # make users face directory if it doesn't exist
        if not os.path.exists(self.user_faces_dir):
            os.makedirs('faces')
        
        # make the csv file to store the face encodings
        if not os.path.exists('face_encodings.csv'):
            f = open('face_encodings.csv', 'w')
            f.close()


    def padRectangle(self, x, y, w, h, size=500):
        ''' Pad the rectangle around the face to be 500x500.'''

        width_padding = size - w
        height_padding = size - h

        return (x, y, w + width_padding, h + height_padding)


    def getFaceCoords(self, gray_frame):
        ''' Outlines the face in the frame if it exists.
            Return the coordinates of rectangle if face exists.'''
        
        # return coordinates
        coordinates = None

        # get the faces
        faces = self.face_classifier.detectMultiScale(gray_frame, 1.3, 5)

        # get rectangle around faces
        for (x, y, w, h) in faces:
            coordinates = self.padRectangle(x, y, w, h)
        
        return coordinates


    def filterImageBySize(self, image):
        ''' Return True if the size of the image passes the threshold specifications.'''

        if (not image or not self.img_sizes):
            return False
        
        width, height = self.img_sizes

        return image.size[0] >= width and image.size[1] >= height


    def getFaceFrame(self, frame):
        ''' Get the face frame from the frame.'''

        face_frame = None

        # convert the frame to RGB instead of BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # get face coordinates
        coords = self.getFaceCoords(gray)

        # get the face coordinates
        if (coords and self.filterImageBySize(Image.fromarray(frame))):
            x, y, w, h = coords
            face_frame = gray[y:y+h, x:x+w]
        
        return face_frame, coords


    def getFaces(self):
        ''' Capture the faces from the camera feed and save them to the faces directory.'''

        print("Getting faces...")

        # number of current faces stored
        num_faces = 0

        # loop through the camera feed
        while num_faces < self.max_faces:
            # read the camera
            ret, frame = self.camera.read()
        
            if not ret:
                break

            # get the face frame
            face_frame, _ = self.getFaceFrame(frame)

            # save the face frame if it exists
            if (face_frame is not None):
                num_faces += 1
                cv2.imwrite(f'{USER_FACES_DIR}/user_{num_faces}.jpg', face_frame)
            
            # stop the camera after 10 faces
            if (num_faces >= self.max_faces):
                break
            
            # display the frame
            cv2.imshow('Face Feed', frame)


    def writeFaceEncodings(self, csv_file_path='face_encodings.csv'):
        ''' Write the face encodings to the csv file.'''

        print("Writing face encodings to csv file...")

        # open the csv file
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            fields = ['file_name', 'is_user', 'path']
            writer.writerow(fields)

            # loop through the user faces
            for i in range(1, self.max_faces + 1):
                face_path = f'{self.user_faces_dir}/user_{i}.jpg'

                # write the face encoding to the csv file
                writer.writerow([f'user_{i}.jpg', 1, face_path])
            
            # loop through the non-user faces
            directory = os.fsencode(self.test_faces_dir)
                
            for file in os.listdir(directory):
                filename = os.fsdecode(file)

                if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
                    file_path = os.path.join(directory.decode(), filename)
                    writer.writerow([filename, 0, file_path])


            f.close()
        print("Wrote face encodings to csv file!")

    # main function
    def generateData(self):
        ''' Main function to generate face data.'''

        # initialize the files/dirs needed
        self.initializeFiles()

        # get faces from the camera feed
        self.getFaces()
        print("Got all faces!")
        
        # write the face encodings to the csv file
        self.writeFaceEncodings()

        self.camera.release()
        cv2.destroyAllWindows()

        print("Finished Generating Data!")


# testing
if __name__ == '__main__':
    face_detector = FaceDetector()
    face_detector.generateData()
