
# Some Computer Vision Ideas I have:

Note: Tested and built on mac. Windows and Linux support may need to be configured, especially with file naming.

## 1. Finger Painting

### Description:

As an introduction to computer vision, I created a finger-painting Python application. Users can use their index finger to paint on a blank canvas, switching between defined colors. Currently, the colors supported are Red, Blue, Green, Orange, Yellow, Pink, Purple, Gray, and White. Feel free to add more colors with their respective RGB values if you wish.

Python Libraries:
- NumPy
- cv2
- mediapipe

Controls:
- A: Previous Color
- D: Next Color
- Enter: Save Image
- Esc: Quit
- Space: Erasor Mode
- P: Clear Image (to white)
- Drawing: Use your index finger!


https://github.com/GlassesNoGlasses/ComputerVision/assets/59126714/2349b52e-96b6-4274-a562-cd880a4eb056


![drawing](https://github.com/GlassesNoGlasses/ComputerVision/assets/59126714/2476ff8d-e3cb-4f8a-9632-aea15e441957)



## 2. Face Recognition

### Description:

A cool little algorithm that detects the user's face and then trains a machine learning model to recognize the user's face using laptop cameras. The detection and recognition models were based on the libraries of OpenCV with configurations. The current recognition model is cv2.LBPHFaceRecognizer().

On the algorithm: the main files are `detector.py`, `trainer.py`, and `recognizer.py`.

`detector.py`: captures `MAX_FACES` number of images from the user using the computer's camera, filters and processes images into valid model inputs (i.e. size, color, normalization), then builds a .csv file, `face_encodings.csv` on the user's images as well as testing images saved in `./test_faces` directory.

`trainer.py`: currently trains the cv2.LBPHFaceRecognizer() based on `face_encodings.csv` using NumPy and Pandas. Label 1 for the user's face, 0 if not.

`recognizer.py`: utilizes the trained model from `trainer.py` and computer camera, capturing new user images and predicting if the image captured on camera is of the user or not. Currently has an 80% threshold for accuracy to verify the user.

On Local Binary Patterns Histograms (LBPH): A quick read of how the cv2.LBPHFaceRecognizer() model works can be found here: https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html#tutorial_face_lbph


Python Libraries:
- NumPy
- Pandas
- cv2
- PIL
- os

How to Run: Install the above libraries, have a working camera, then run `recognizer.py`.

Future Prospects: Building an Eigenface or Fisherface model from scratch utilizing Scikit-Learn and other python libraries.
