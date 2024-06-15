
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

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

# images
DRAW_IMG = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype = np.uint8) * 255
INSTUCTION_IMG = np.ones((350, 250, 3), dtype = np.uint8) * 255

# image drawing
line_coordinates = {'start': (-1, -1), 'end': (-1, -1)}

# colors
GREEN = {'rgb': (0, 255, 0), 'name': 'Green'}
BLUE = {'rgb': (255, 0, 0), 'name': 'Blue'}
YELLOW = {'rgb': (0, 255, 255), 'name': 'Yellow'}
ORANGE = {'rgb': (0, 165, 255), 'name': 'Orange'}
PURPLE = {'rgb': (128, 0, 128), 'name': 'Purple'}
PINK = {'rgb': (203, 192, 255), 'name': 'Pink'}
GRAY = {'rgb': (128, 128, 128), 'name': 'Gray'}
WHITE = {'rgb': (255, 255, 255), 'name': 'White'}
RED = {'rgb': (0, 0, 255), 'name': 'Red'}
BLACK = {'rgb': (0, 0, 0), 'name': 'Black'}

COLORS = [GRAY, RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK, WHITE]
ERASOR = WHITE

# color selection logic
NUM_COLORS = len(COLORS)
chosen_color_index = 0
chosen_color = COLORS[chosen_color_index]
prev_color = COLORS[(chosen_color_index - 1) % NUM_COLORS]['name']
next_color = COLORS[(chosen_color_index + 1) % NUM_COLORS]['name']
is_erasor = False

# cv2 keyboard input
A_KEY = 97
D_KEY = 100
P_KEY = 112
SPACE = 32
ESC = 27
ENTER = 13

# feedback
color_feedback = "Color: " + chosen_color['name']
controls_feedback = {'prev': "(A): " + prev_color,
                     'next': "(D): " + next_color, 'clear': "Clear: (P)"}
erasor_feedback = "Erasor: " + "On" if is_erasor else "OFF"

while cap.isOpened():
    # get the key pressed
    key = cv2.waitKey(33)

    # key press logic
    if key == ESC:
        break
    elif key == SPACE:
        is_erasor = not is_erasor
    elif key == A_KEY:
        chosen_color_index = (chosen_color_index - 1) % NUM_COLORS
    elif key == D_KEY:
        chosen_color_index = (chosen_color_index + 1) % NUM_COLORS
    elif key == P_KEY:
        DRAW_IMG = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype = np.uint8) * 255
    elif key == ENTER:
        cv2.imwrite('drawing.jpg', DRAW_IMG)
    
    # update the colors
    chosen_color = COLORS[chosen_color_index]
    prev_color = COLORS[(chosen_color_index - 1) % NUM_COLORS]['name']
    next_color = COLORS[(chosen_color_index + 1) % NUM_COLORS]['name']

    # update the feedback
    color_feedback = "Color: " + chosen_color['name']
    controls_feedback['prev'] = "(A): " + prev_color
    controls_feedback['next'] = "(D): " + next_color
    erasor_feedback = "Erasor: " + "On" if is_erasor else "OFF"

    # read the camera
    ret, frame = cap.read()

    if (not ret):
        break

    # convert the frame to RGB instead of BGR
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(RGB_frame)

    inst_img = INSTUCTION_IMG.copy()

    # if hands are detected, draw the landmarks
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # obtain index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[8]
            index_finger_coords = (abs(int(index_finger_tip.x * FRAME_WIDTH) - FRAME_WIDTH), abs(int(index_finger_tip.y * FRAME_HEIGHT)))
            
            # update line coordinates
            if line_coordinates['start'] != (-1, -1):
                line_coordinates['start'] = line_coordinates['end']
                line_coordinates['end'] = index_finger_coords
            else:
                line_coordinates['start'] = index_finger_coords

            # draw line
            if not is_erasor:
                DRAW_IMG = cv2.line(DRAW_IMG, line_coordinates['start'], line_coordinates['end'], chosen_color['rgb'], 2)
            else:
                DRAW_IMG = cv2.line(DRAW_IMG, line_coordinates['start'], line_coordinates['end'], ERASOR['rgb'], 8)

    # user feedback
    cv2.putText(inst_img, color_feedback,
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK['rgb'], 2)
    cv2.putText(inst_img, erasor_feedback,
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK['rgb'], 2)
    cv2.putText(inst_img, controls_feedback['prev'],
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK['rgb'], 2)
    cv2.putText(inst_img, controls_feedback['next'],
                (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK['rgb'], 2)
    cv2.putText(inst_img, controls_feedback['clear'],
                (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK['rgb'], 2)
    cv2.putText(inst_img, "Save: (ENTER)",
                (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK['rgb'], 2)
    
    # display the frame
    cv2.imshow('Instructions', inst_img)
    cv2.imshow('Drawing', DRAW_IMG)

cap.release()
cv2.destroyAllWindows()
