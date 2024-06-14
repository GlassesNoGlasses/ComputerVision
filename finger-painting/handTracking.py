
import cv2
import numpy as np
import mediapipe as mp

# frame dimensions
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# cv2 cameria object
cap = cv2.VideoCapture(1)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

# mediapipe hands object
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# image
BLACK_IMG = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype = np.uint8)


while cap.isOpened():

    # read the camera
    ret, frame = cap.read()

    if (not ret):
        break

    black_img = BLACK_IMG.copy()

    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(RGB_frame)

    # if hands are detected, draw the landmarks
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            mpDraw.draw_landmarks(black_img, hand_landmarks, mpHands.HAND_CONNECTIONS)
    
    # display the frame
    cv2.imshow('frame', frame)
    cv2.imshow('black_img', black_img)


    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
