
# import libraries
import cv2
import numpy as np

GREEN_COLOR = (0, 255, 0)
WHITE_COLOR = (255, 255, 255)

camera = cv2.VideoCapture(1)

ret, frame1 = camera.read()
ret, frame2 = camera.read()

while camera.isOpened():
    # get difference between two frames
    diff = cv2.absdiff(frame1, frame2)

    # grayscale the difference for easier input size
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # blur the image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image to get the binary image; seperates the background from the foreground
    _, threshold = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # dilate the image to fill in the holes and brighten the image
    dilated = cv2.dilate(threshold, None, iterations=3)

    # find the contours of the image based on dilations; connect pixels of the same color
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # show contours on the frame
    cv2.drawContours(frame1, contours, -1, GREEN_COLOR, 2)

    # black_img = np.zeros((1000, 1000, 3), dtype = np.uint8)
    # cv2.drawContours(black_img, contours, -1, WHITE_COLOR, 2)

    # display the frame
    cv2.imshow('feed', frame1)
    # cv2.imshow('black', black_img)

    # set frame1 to frame2 and frame2 to the next frame
    frame1 = frame2
    ret, frame2 = camera.read()

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
