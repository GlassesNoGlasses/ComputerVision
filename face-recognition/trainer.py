
import cv2
import numpy as np
import pandas as pd
from PIL import Image


# face recognizer class
class LBPH():

    def __init__(self, csv_path='face_encodings.csv', specs=(1, 8, 8, 8)):
        ''' Initialize the face recognizer with the face detector.'''

        radius, neighbors, grid_x, grid_y = specs

        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer.create(
                radius=radius,
                neighbors=neighbors,
                grid_x=grid_x,
                grid_y=grid_y)
    
            self.face_csv = pd.read_csv(csv_path)
        except (cv2.error):
            print('Error initializing the face recognizer. Check OpenCV is installed.')
        except (FileNotFoundError):
            print('Error reading the face encodings file. Please specify the correct csv path.')
        
    def getFaceData(self):
        ''' Get the face images and labels from the csv file.'''

        # get the face paths and labels from the csv file
        face_paths = self.face_csv['path'].tolist()
        face_labels = self.face_csv['is_user'].tolist()

        faces = []
        labels = []

        # loop through the face paths and labels
        for path, label in zip(face_paths, face_labels):
            # open image in greyscale
            face_image = Image.open(path).convert('L')

            # convert image to numpy array
            face_np = np.array(face_image, 'uint8')

            # append data to lists
            faces.append(face_np)
            labels.append(label)
        
        return faces, labels


    def trainRecognizer(self):
        ''' Train the face recognizer with the face encodings of csv file.'''

        # get the face data
        faces, labels = self.getFaceData()

        # train the recognizer
        self.face_recognizer.train(faces, np.array(labels))
    

    def recognizeFace(self, gray_image):
        ''' Recognize the face in the image.'''

        prediction = self.face_recognizer.predict(gray_image)
        print('Recognizing face...')
        print(prediction)

