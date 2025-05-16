from collections import deque # Import deque for efficiently managing a fixed-length queue
import cv2 as cv


class CvFpsCalc(object):
    def __init__(self, bufferLen=1):
        """
        Initialize the FPS calculator.
        :param bufferLen: The number of recent frame times to average for FPS calculation.
        """
        self._startTick = cv.getTickCount() # Get the initial tick count (high-resolution timer)
        self._freq = 1000.0 / cv.getTickFrequency() # Convert tick frequency to milliseconds
        self._difftimes = deque(maxlen=bufferLen) # Create a deque to store frame times, limited to `bufferLen`

    def get(self):
        """
        Calculate and return the current FPS.
        :return: The calculated FPS as a rounded float.
        """
        currentTick = cv.getTickCount() # Get the current tick count
        differentTime = (currentTick - self._startTick) * self._freq  #Calculate the time difference in milliseconds
        self._startTick = currentTick # Update the start tick to the current tick

        self._difftimes.append(differentTime) # Add the time difference to the deque

        # Calculate FPS as 1000 ms divided by the average frame time
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        roundedFps = round(fps, 1)

        return roundedFps
