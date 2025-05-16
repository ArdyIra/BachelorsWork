import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2 as cv
from handGestureRecognition import HandGestureRecognition

#launch tests with python -m unittest discover -s tests

class TestHandGestureRecognition(unittest.TestCase):
    def setUp(self):
        """Set up the HandGestureRecognition instance for testing."""
        self.handRecog = HandGestureRecognition(
            device=0,
            width=640,
            height=480,
            useStaticImgMode=True,
            minDetConf=0.7,
            minTrackConf=0.5,
            appMode="standalone",
        )

    @patch('cv2.VideoCapture')
    def test_CameraInit(self, mock_videoCapture):
        """Test if the camera is initialized correctly."""
        mock_videoCapture.return_value.isOpened.return_value = True
        self.handRecog.cap = mock_videoCapture(0)
        self.assertTrue(self.handRecog.cap.isOpened())

    def test_landmarkPreproc(self):
        """Test the preprocessing of landmarks."""
        landmarks = [[100, 200], [150, 250], [200, 300]]
        processed = self.handRecog.landmarkPreprocessing(landmarks)
        self.assertEqual(len(processed), len(landmarks) * 2)  # Flattened list
        self.assertAlmostEqual(max(map(abs, processed)), 1.0)  # Normalized

    @patch('cv2.VideoCapture.read')
    def test_processFrameNoFrame(self, mock_read):
        """Test process_frame when no frame is captured."""
        mock_read.return_value = (False, None)
        result = self.handRecog.processFrame()
        self.assertFalse(result)

    @patch('cv2.VideoCapture.read')
    @patch('mediapipe.solutions.hands.Hands.process')
    def test_processFrameWithHand(self, mock_process, mock_read):
        """Test process_frame when a hand is detected."""
        # Mock a frame
        mock_read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        # Mock Mediapipe results
        mock_landmark = MagicMock()
        mock_landmark.landmark = [MagicMock(x=0.5, y=0.5), MagicMock(x=0.6, y=0.6)]
        mock_process.return_value.multi_hand_landmarks = [MagicMock()]
        mock_process.return_value.multi_handedness = [MagicMock()]

        # Run the method
        result = self.handRecog.processFrame()
        self.assertTrue(result)

    def test_simulate_typing(self):
        """Test the simulate_typing method."""
        self.handRecog.keyboard = MagicMock()
        self.handRecog.simulateTyping("A")
        self.handRecog.keyboard.press.assert_called_with("a")
        self.handRecog.keyboard.release.assert_called_with("a")

    @patch('csv.writer')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_csvLogging(self, mock_open, mock_csvWriter):
        """Test CSV logging."""
        self.handRecog.mode = 1
        self.handRecog.csvLogging(1, [0.1, 0.2, 0.3])
        mock_open.assert_called_with('utils/model/landPoints.csv', 'a', newline="")
        mock_csvWriter.return_value.writerow.assert_called_with([1, 0.1, 0.2, 0.3])

    def test_findBoundRect(self):
        """Test finding the bounding rectangle."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = MagicMock()
        landmarks.landmark = [MagicMock(x=0.5, y=0.5), MagicMock(x=0.6, y=0.6)]
        brect = self.handRecog.findBoundRect(image, landmarks)
        self.assertEqual(len(brect), 4)  # [x, y, x+w, y+h]

    def test_drawLandmarks(self):
        """Test drawing landmarks on an image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = [[100, 200], [150, 250], [200, 300]]
        resultImg = self.handRecog.drawLandmarks(image, landmarks)
        self.assertEqual(resultImg.shape, image.shape)  # Ensure image size is unchanged

    def test_drawInfoText(self):
        """Test drawing info text on an image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        brect = [10, 10, 100, 100]
        handedness = MagicMock()
        handedness.classification = [MagicMock(label="Right")]
        resultImg = self.handRecog.drawInfoText(image, brect, handedness, "Gesture")
        self.assertEqual(resultImg.shape, image.shape)  # Ensure image size is unchanged


if __name__ == '__main__':
    unittest.main()