
import time
import numpy as np
import pandas as pd
import cv2
import pickle
from detector import FaceDetector
from sklearn import svm
from sklearn import decomposition
from sklearn import metrics
from PIL import Image
from sklearn.model_selection import train_test_split
from detector import IMG_SIZES

RANDOM_STATE = 42
ESC = 27

class Eigenface():

    def __init__(self, csv_path='face_encodings.csv') -> None:

        try:
            self.face_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print('Error reading the face encodings file. Please specify the correct csv path.')
        

    def getFaceData(self):
        ''' Get the face images and labels from the csv file.
        Returns: faces, labels as numpy arrays.'''

        # get the face paths and labels from the csv file
        face_paths = self.face_df['path'].tolist()
        face_labels = self.face_df['is_user'].tolist()

        faces = []
        labels = []

        # loop through the face paths and labels
        for path, label in zip(face_paths, face_labels):
            # open image in greyscale
            face_image = Image.open(path).convert('L')

            # resize and save image to path
            resized_image = face_image.resize(IMG_SIZES, Image.Resampling.LANCZOS)

            # convert image to numpy array
            face_np = np.array(resized_image, 'uint8')

            # append data to lists
            faces.append(face_np)
            labels.append(int(label))
        
        return faces, labels
    

    def pcaInit(self, num_components=30):
        ''' Train the eigenface model.'''

        # get the face data
        faces, labels = self.getFaceData()

        # flatten paces to 1D array
        pcaFaces = np.array([f.flatten() for f in faces])

        # construct our training and testing split
        split = train_test_split(faces, pcaFaces, labels, test_size=0.25,
            stratify=labels, random_state=RANDOM_STATE)
    
        origTrain, origTest, trainX, testX, trainY, testY = split

        self.trainX = trainX
        self.testX = testX
        self.trainY = trainY
        self.testY = testY

        # create pca model
        self.pca = decomposition.PCA(n_components=num_components, whiten=True, svd_solver='randomized')

        print("[INFO] Configuring pca...")

        # train pca model
        start = time.time()
        self.trainX = self.pca.fit_transform(self.trainX)
        end = time.time()

        print(f'Training took {end - start} seconds.')

        # reshape the components
        images = []

        for component in self.pca.components_:
            # reshape component to 2D array of (500, 500)
            component = component.reshape(IMG_SIZES)
            component = np.dstack([component.astype("uint8")] * 3)
            # image = Image.fromarray(component).convert('L')
            images.append(component)
        
        # obtain the mean image
        mean_face = self.pca.mean_.reshape(IMG_SIZES).astype('uint8')
        self.mean_image = mean_face


    def trainEigenface(self):
        ''' Train the eigenface SVC model.'''

        # initialize svc model
        self.svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.001, random_state=RANDOM_STATE)

        print("[INFO] Training SVC model...")

        # train the model

        start = time.time()
        self.trainX = self.svc.fit(self.trainX, self.trainY)
        end = time.time()

        print(f'Training took {end - start} seconds.')

        # evaluate the model on test data
        print("[INFO] evaluating model...")

        predictions = self.svc.predict(self.pca.transform(self.testX))
        print(metrics.classification_report(self.testY, predictions,
            target_names=['imposter', 'user']))
    
        # save the model
        with open('eigenface_model.pkl','wb') as f:
            pickle.dump(self.svc, f)


    def predictEigenface(self):
        ''' Predict the face using the eigenface model.'''

        camera = cv2.VideoCapture(1)
        detector = FaceDetector()

        while camera.isOpened():

            # get key press if exists
            key = cv2.waitKey(33)

            if key == ESC:
                break

            # read frame from camera
            ret, frame = camera.read()

            if not ret:
                print('Error reading frame.')
                break

            # get the face frame
            face_image, coords = detector.getFaceFrame(frame)

            # could not detect face
            if face_image is None or face_image.shape != IMG_SIZES or coords is None:
                continue

            # get face coords
            x, y, w, h = coords

            # draw rectangle on face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # convert face image to numpy array
            pca_image = face_image.flatten()

            # apply PCA dimension reduction and predict
            pca_frame = self.pca.transform(pca_image.reshape(1, -1))
            prediction = self.svc.predict(pca_frame)

            if prediction[0] == 1:
                cv2.putText(frame, 'Person: User', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            else:
                cv2.putText(frame, 'Person: Imposter', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow('Eigenface', frame)
        
        camera.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    eigenface = Eigenface()
    eigenface.pcaInit()
    eigenface.trainEigenface()
    # eigenface.predictEigenface()
