import csv #for csv file handling
import copy #for deep copy of images
import itertools #for flattening lists

import cv2 as cv #for image processing
import numpy as np #for numerical operations
import mediapipe as mp #for hand detection and tracking

from utils.cvFpsCalc import CvFpsCalc #for calculating frames per second
from utils.model.landPointClassifier import LandPointClassifier #for classifying hand gestures
from pynput.keyboard import Controller #for keyboard input handling

import threading
import time

import os
import warnings
import absl.logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress Mediapipe logs
absl.logging.set_verbosity(absl.logging.ERROR)

# Suppress Python warnings
warnings.filterwarnings("ignore")


class HandGestureRecognition:
    def __init__(self, device=0, width=960, height=540, 
                 useStaticImgMode=False,
                 minDetConf=0.7, 
                 minTrackConf=0.5, 
                 appMode="standalone",
                 updateTypedKeys_callback=None, 
                 inputSpeed_callback=None, 
                 gestureHoldThreshold=20,
                 ):
        
        # Initialize parameters
        self.device = device
        self.width = width
        self.height = height
        self.useStaticImgMode = useStaticImgMode
        self.minDetConf = minDetConf
        self.minTrackConf = minTrackConf
        self.appMode = appMode # "embedded" for UI, "standalone" OpenCV window
        self.updateTypedKeys_callback = updateTypedKeys_callback  # Callback for updating typed keys
        self.inputSpeed_callback = inputSpeed_callback  # Callback to get the input speed

        self.keyboard = Controller()  # Initialize the keyboard controller
        self.gestureHoldCount = 0  # Track how many frames the gesture has been held
        self.gestureHoldThreshold = gestureHoldThreshold  # Number of frames required to confirm a gesture
        self.lastGesture = None  # Track the last gesture to avoid repeated typing

        # Initialize camera
        self.cap = cv.VideoCapture(self.device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

        #Multithreading for camera capture
        self.frame = None
        self.ret = False
        self.stopped = False
        self.captureLock = threading.Lock()
        self.captureThread = threading.Thread(target=self._updateFrame, daemon=True)
        self.captureThread.start()

        # Initialize Mediapipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.useStaticImgMode,
            max_num_hands=1,
            min_detection_confidence=self.minDetConf,
            min_tracking_confidence=self.minTrackConf,
        )

        # Initialize keypoint classifier
        self.landpointClassifier = LandPointClassifier()

        if self.appMode == "standalone":
            self.loggingIncrement = 0  # Initialize logging increment for standalone mode

        # Load keypoint labels
        with open('utils/model/landPointClassifier_labels.csv', encoding='utf-8-sig') as f:
            landpointClassLabels = csv.reader(f)
            self.landpointClassLabels = [row[0] for row in landpointClassLabels]

        # Initialize FPS calculator
        self.cvFpsCalc = CvFpsCalc(bufferLen=10)

        # Initialize mode
        self.mode = 0 # Default mode (e.g., 0 for normal, 1 for logging)

    def _updateFrame(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.captureLock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.001)  # Small sleep to reduce CPU usage

    def modeSelection(self, keyPress):
        if self.appMode == "embedded":
            return -1  # In embedded mode, return -1 to avoid key press handling
        
        numba = -1
        if 48 <= keyPress <= 57:  # 0-9
            numba = keyPress - 48
        if keyPress == 100:  # 'D'efault
            self.mode = 0
        if keyPress == 115:  # 'S'ave
            self.mode = 1
        if keyPress == 43:
            self.loggingIncrement += 10
        if keyPress == 45:
            if self.loggingIncrement > 9:
                self.loggingIncrement -= 10
        return numba

    def processFrame(self):
        fps = self.cvFpsCalc.get()
        key = cv.waitKey(1)
        if key == 27:  # ESC
            return False  # Exit loop
        number = self.modeSelection(key)   

        #New
        with self.captureLock:
            ret = self.ret
            image = self.frame.copy() if self.frame is not None else None
        if not ret or image is None:
            return False

        image = cv.flip(image, 1)  # Mirror display
        processedImage = copy.deepcopy(image)

        # Process image
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Convert BGR to RGB
        image.flags.writeable = False # Disable writing to the image
        results = self.hands.process(image) # Process the image with Mediapipe Hands
        image.flags.writeable = True # Re-enable writing to the image
        #question: why is this needed? 

        # Process hand landmarks
        if results.multi_hand_landmarks is not None:
            for handLands, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                bRect = self.findBoundRect(processedImage, handLands)
                landmark_list = self.findLandmarks(processedImage, handLands)
                preprocessedLand_list = self.landmarkPreprocessing(landmark_list)
                if not preprocessedLand_list:
                    continue # Skip if no landmarks found
                self.csvLogging(number, preprocessedLand_list)
                handSignId = self.landpointClassifier(preprocessedLand_list)

                if self.appMode == "embedded":
                    # Get the gesture label
                    gesture = self.landpointClassLabels[handSignId]

                    # Adjust the gesture hold threshold based on the slider value
                    if self.inputSpeed_callback:
                        self.gestureHoldThreshold = self.inputSpeed_callback()
                    else:
                        self.gestureHoldThreshold = 1.0  # Default value if callback is not provided

                    # Convert seconds to frame delay based on actual FPS
                    frameDelayThreshold = int(self.gestureHoldThreshold * fps)

                    # Check if the gesture is the same as the last one
                    if gesture == self.lastGesture:
                        self.gestureHoldCount += 1
                        if self.gestureHoldCount >= frameDelayThreshold:
                            self.simulateTyping(gesture)
                            self.gestureHoldCount = 0  # Reset the hold count after typing
                    else:
                        self.lastGesture = gesture
                        self.gestureHoldCount = 1  # Reset the hold count for the new gesture
                
                # Draw on the image
                processedImage = self.drawBoundRect(processedImage, bRect)
                processedImage = self.drawLandmarks(processedImage, landmark_list)
                processedImage = self.drawInfoText(processedImage, bRect, handedness, self.landpointClassLabels[handSignId])

        processedImage = self.drawModeInfo(processedImage, fps, self.mode, number)

        if self.appMode == "standalone":
            cv.imshow('Hand Gesture Recognition', processedImage)  # Show OpenCV window
            return True
        elif self.appMode == "embedded":
            return processedImage  # Return the processed image for embedding
        

    def simulateTyping(self, gesture):
        # Map gestures to keyboard inputs
        gesture2Key = {
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "E": "e",
            "F": "f",
            "G": "g",
            "H": "h",
            "I": "i",
            "J": "j",
            "K": "k",
            "L": "l",
            "M": "m",
            "N": "n",
            "O": "o",
            "P": "p",
            "Q": "q",
            "R": "r",
            "S": "s",
            "T": "t",
            "U": "u",
            "V": "v",
            "W": "w",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "Space": "space",
            "Enter": "enter",
            "Backspace": "backspace"
        }

        # Simulate the key press
        if gesture in gesture2Key:
            key = gesture2Key[gesture]
            if key == "space":
                self.keyboard.press(" ")
                self.keyboard.release(" ")
            elif key == "enter":
                self.keyboard.press("\n")
                self.keyboard.release("\n")
            elif key == "backspace":
                self.keyboard.press("\b")
                self.keyboard.release("\b")
            else:
                self.keyboard.press(key)
                self.keyboard.release(key)

            # Update the UI with the typed key
            if self.updateTypedKeys_callback:
                self.updateTypedKeys_callback("flash_border")


    def run(self):
        while True:
            with self.captureLock:
                if self.ret and self.frame is not None:
                    break
            time.sleep(0.001)  # Small sleep to reduce CPU usage


        while True:
            if not self.processFrame():
                break
        
        # Release resources
        self.stopped = True
        self.captureThread.join()
        self.cap.release()
        cv.destroyAllWindows()

    # Utility functions (unchanged from original code)
    def findBoundRect(self, image, landmarks):
        imgWidth, imgHeight = image.shape[1], image.shape[0]
        landmark_arr = np.empty((0, 2), int)
        for _, landmark in enumerate(landmarks.landmark):
            landX = min(int(landmark.x * imgWidth), imgWidth - 1) # Normalize x coordinate
            landY = min(int(landmark.y * imgHeight), imgHeight - 1) # Normalize y coordinate
            landPoint = [np.array((landX, landY))]
            landmark_arr = np.append(landmark_arr, landPoint, axis=0)
        x, y, w, h = cv.boundingRect(landmark_arr)
        return [x, y, x + w, y + h]

    def findLandmarks(self, image, landmarks):
        imgWidth, imgHeight = image.shape[1], image.shape[0]
        landPoint = []
        for _, landmark in enumerate(landmarks.landmark):
            landX = min(int(landmark.x * imgWidth), imgWidth - 1)
            landY = min(int(landmark.y * imgHeight), imgHeight - 1)
            landPoint.append([landX, landY])
        return landPoint

    # Convert landmarks to relative coordinates
    def landmarkPreprocessing(self, landmark_list):
        tempLand_list = copy.deepcopy(landmark_list)
        if not tempLand_list:
            return [] # Return empty list if no landmarks found
        baseX, baseY = tempLand_list[0]
        for index, land_point in enumerate(tempLand_list):
            tempLand_list[index][0] -= baseX
            tempLand_list[index][1] -= baseY
        tempLand_list = list(itertools.chain.from_iterable(tempLand_list))
        maxVal = max(map(abs, tempLand_list))
        return [n / maxVal for n in tempLand_list]

    def csvLogging(self, number, land_list):
        if self.appMode == "embedded":
            return  # Skip CSV logging in embedded mode
        
        if self.mode == 1 and (0 <= number <= 9):
            csvPath = 'utils/model/landPoints.csv'
            with open(csvPath, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number + self.loggingIncrement, *land_list])

    def drawLandmarks(self, image, landPoint):
        if len(landPoint) > 0:
            # Define connections for fingers and palm
            connections = [
                (2, 3), (3, 4),  # Thumb
                (5, 6), (6, 7), (7, 8),  # Index finger
                (9, 10), (10, 11), (11, 12),  # Middle finger
                (13, 14), (14, 15), (15, 16),  # Ring finger
                (17, 18), (18, 19), (19, 20),  # Little finger
                (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)  # Palm
            ]

            # Draw lines
            for start, end in connections:
                if start < len(landPoint) and end < len(landPoint):  # Check for valid indices
                    cv.line(image, tuple(landPoint[start]), tuple(landPoint[end]), (0, 0, 0), 6)
                    cv.line(image, tuple(landPoint[start]), tuple(landPoint[end]), (255, 255, 255), 2)

            # Draw key points
            for index, landmark in enumerate(landPoint):
                if index in [4, 8, 12, 16, 20]:  # Fingertips
                    color = (0, 165, 255)  # Orange in BGR
                    radius = 8
                else:  # Regular landmarks
                    color = (208, 224, 64)  # Teal in BGR
                    radius = 5
                cv.circle(image, (landmark[0], landmark[1]), radius, color, -1)
                cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

        return image


    def drawBoundRect(self, image, brect):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
        return image

    def drawInfoText(self, image, brect, handedness, handSignInfo):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
        infoText = handedness.classification[0].label[0:]
        if handSignInfo != "":
            infoText += ':' + handSignInfo
        cv.putText(image, infoText, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        return image

    def drawModeInfo(self, image, fps, mode, number):
        cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA) # Black shadow
        cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA) # White text
        modeString = ['Logging Landmark Points']
        if 1 <= mode <= 2:
            cv.putText(image, f"MODE: {modeString[mode - 1]} - increment: {self.loggingIncrement}", (10, 90), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
            cv.putText(image, f"MODE: {modeString[mode - 1]} - increment: {self.loggingIncrement}", (10, 90), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            if 0 <= number <= 9:
                cv.putText(image, f"NUM: {number}", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
                cv.putText(image, f"NUM: {number}", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        return image


if __name__ == '__main__':
    app = HandGestureRecognition()
    app.run()