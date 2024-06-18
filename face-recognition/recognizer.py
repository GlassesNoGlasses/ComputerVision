
import cv2
from detector import FaceDetector
from detector import IMG_SIZES
from trainer import LBPH
from eigenface import Eigenface

# initialize the camera
camera = cv2.VideoCapture(1)

# classifier
classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# keybinds
ESC = 27

# main function
def main(mode='LBPH'):
    ''' Main function to recognize faces.'''

    # initialize the face detector
    face_detector = FaceDetector()

    # generate the face data
    face_detector.generateData()

    # initialize the face recognizer
    face_recognizer = LBPH() if mode == 'LBPH' else Eigenface()

    # train the face recognizer
    face_recognizer.trainRecognizer()

    # loop through the camera feed
    while camera.isOpened():

        # break the loop if ESC key is pressed
        if cv2.waitKey(33) & 0xFF == ESC:
            break

        ret, frame = camera.read()

        if not ret:
            break
        
        # get face frame to detect
        face_frame, coords = face_detector.getFaceFrame(frame)

        if face_frame is None or coords is None or face_frame.shape != IMG_SIZES:
            continue

        x, y, w, h = coords
        success = False

        # draw the rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        match mode:
            case 'LBPH':
                # predict the face
                label, loss = face_recognizer.face_recognizer.predict(face_frame)

                # display the label and confidence
                print(f'Label: {label}, Loss: {loss}')

                # 80% accurate
                if (loss < 20):
                    # user feedback
                    success = True
                    cv2.putText(frame, "Person Detected: User",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    # user feedback
                    cv2.putText(frame, "Person Detected: Unknown",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            case 'Eigenface':
                # convert face image to numpy array
                pca_image = face_frame.flatten()

                # apply PCA dimension reduction and predict
                pca_frame = face_recognizer.pca.transform(pca_image.reshape(1, -1))
                prediction = face_recognizer.face_recognizer.predict(pca_frame)

                if prediction[0] == 1:
                    success = True

        # user feedback
        if success:
            cv2.putText(frame, "Person Detected: User",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            # user feedback
            cv2.putText(frame, "Person Detected: Unknown",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            
        # display the frame
        cv2.imshow('Face Feed', frame)
    
    # release the camera
    camera.release()
    cv2.destroyAllWindows()

    print("Exiting...")



if __name__ == '__main__':
    mode = input('Enter the mode (LBPH or Eigenface): ')

    while mode not in ['LBPH', 'Eigenface']:
        mode = input('Enter the mode (LBPH or Eigenface): ')

    main(mode)
