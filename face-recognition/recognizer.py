
import cv2
from detector import FaceDetector
from trainer import FaceRecognizer

# initialize the camera
camera = cv2.VideoCapture(1)

# classifier
classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# keybinds
ESC = 27

# main function
def main():
    ''' Main function to recognize faces.'''

    # initialize the face detector
    face_detector = FaceDetector()

    # generate the face data
    face_detector.generateData()

    # initialize the face recognizer
    face_recognizer = FaceRecognizer()

    # train the face recognizer
    face_recognizer.trainRecognizer()

    # loop through the camera feed
    while camera.isOpened():
        ret, frame = camera.read()

        if not ret:
            break
        
        # get face frame to detect
        face_frame, coords = face_detector.getFaceFrame(frame)

        if face_frame is None or coords is None:
            continue

        x, y, w, h = coords

        # predict the face
        label, loss = face_recognizer.face_recognizer.predict(face_frame)

        # display the label and confidence
        print(f'Label: {label}, Loss: {loss}')

        # draw the rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 80% accurate
        if (loss < 20):
            # user feedback
            cv2.putText(frame, "Person Detected: User",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            # user feedback
            cv2.putText(frame, "Person Detected: Unknown",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
        # display the frame
        cv2.imshow('Face Feed', frame)
        cv2.imshow('Face Frame', face_frame)

        # break the loop if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == ESC:
            break
    
    # release the camera
    camera.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
