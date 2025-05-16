import csv
#import time
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2 as cv
from PIL import Image, ImageTk
import numpy as np

from handGestureRecognition import HandGestureRecognition  # Import the HandGestureRecognition class
from pygrabber.dshow_graph import FilterGraph  # Import FilterGraph to list camera names
import os  # Import os to work with file paths
import threading  # Import threading for video processing in a separate thread

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")
        self.root.geometry("1280x720")

        # Initialize variables
        self.selectedCam = tk.StringVar()
        self.selectedModel = tk.StringVar()
        self.running = False
        self.handRecog = None

        # Create UI components
        self.createUI()

    def createUI(self):
        # Frame for Camera and Model selection
        selectionFrame = tk.Frame(self.root)
        selectionFrame.pack(pady=10)

        # Camera selection
        cameraLabel = tk.Label(selectionFrame, text="Select Camera:")
        cameraLabel.grid(row=0, column=0, padx=5)
        self.cameraDropdown = ttk.Combobox(selectionFrame, textvariable=self.selectedCam)
        self.cameraDropdown['values'] = self.getCameraList()
        self.cameraDropdown.current(0)
        self.cameraDropdown.grid(row=0, column=1, padx=5)

        # Model selection
        modelLabel = tk.Label(selectionFrame, text="Select Model:")
        modelLabel.grid(row=0, column=2, padx=5)
        self.modelDropdown = ttk.Combobox(selectionFrame, textvariable=self.selectedModel)
        self.modelDropdown['values'] = self.getModelList()
        self.modelDropdown.current(0)
        self.modelDropdown.grid(row=0, column=3, padx=5)

        # Frame for Start and Stop buttons
        buttonFrame = tk.Frame(self.root)
        buttonFrame.pack(pady=10)

        # Start and Stop buttons
        self.startButton = tk.Button(buttonFrame, text="Start", command=self.startRecognition)
        self.startButton.grid(row=0, column=0, padx=10)
        self.stopButton = tk.Button(buttonFrame, text="Stop", command=self.stopRecognition, state=tk.DISABLED)
        self.stopButton.grid(row=0, column=1, padx=10)

        # Frame for the sliders
        sliderFrame = tk.Frame(self.root)
        sliderFrame.pack(pady=10)

        # Input Speed Slider
        sliderLabel = tk.Label(sliderFrame, text="Input Speed (seconds):")
        sliderLabel.grid(row=0, column=0, padx=5)
        self.inputSpeed = tk.IntVar(value=1.0)  # Default speed value
        self.speedSlider = tk.Scale(sliderFrame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.inputSpeed)
        self.speedSlider.grid(row=1, column=0, padx=5)

        # Min Detection Confidence Slider
        detectionLabel = tk.Label(sliderFrame, text="Min Detection Confidence:")
        detectionLabel.grid(row=0, column=1, padx=5)
        self.minDetConf = tk.DoubleVar(value=0.7)  # Default confidence value
        self.detectionSlider = tk.Scale(sliderFrame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                                        variable=self.minDetConf)
        self.detectionSlider.grid(row=1, column=1, padx=5)

        # Min Tracking Confidence Slider
        trackingLabel = tk.Label(sliderFrame, text="Min Tracking Confidence:")
        trackingLabel.grid(row=0, column=2, padx=5)
        self.minTrackConf = tk.DoubleVar(value=0.5)  # Default confidence value
        self.trackingSlider = tk.Scale(sliderFrame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                                        variable=self.minTrackConf)
        self.trackingSlider.grid(row=1, column=2, padx=5)

        # Frame for the video display area with a black border
        self.videoFrame = tk.Frame(self.root, bg="black", highlightthickness=5, highlightbackground="black")
        self.videoFrame.pack(pady=10)
        self.videoLabel = tk.Label(self.videoFrame)
        self.videoLabel.pack()

    def flashBorder(self):
        # Change the border color to red
        self.videoFrame.config(bg="red", highlightbackground="red")

        # Reset the border color to black after 250ms (half a second)
        self.root.after(250, lambda: self.videoFrame.config(bg="black", highlightbackground="black"))

    def updateTypedKeys(self, signal):
        if signal == "flash_border":
            self.flashBorder()

    def getModelList(self):
        # Get the list of available models in the model path directory
        modelPath = 'utils/model/'  # Path to the models
        models = [f for f in os.listdir(modelPath) if f.endswith('.tflite')]  # List .tflite files
        return models if models else ["No Model Found"]

    def getCameraList(self):
        # Detect available cameras and return their names
        graph = FilterGraph()
        cameraList = graph.get_input_devices()  # Get the list of camera names
        return cameraList if cameraList else ["No Camera Found"]

    def startRecognition(self):
        # Start the hand gesture recognition
        if self.selectedCam.get() == "No Camera Found":
            messagebox.showerror("Error", "No camera available!")
            return

        if self.selectedModel.get() == "No Model Found":
            messagebox.showerror("Error", "No model available!")
            return

        # Get the index of the selected camera
        cameraIndex = self.getCameraList().index(self.selectedCam.get())

        # Get the selected model and its corresponding label file
        modelPath = f"utils/model/{self.selectedModel.get()}"
        labelPath = modelPath.replace('.tflite', '_labels.csv')

        # Initialize HandGestureRecognition with the selected model and label file
        self.handRecog = HandGestureRecognition(
            device=cameraIndex, 
            appMode="embedded", 
            updateTypedKeys_callback=self.updateTypedKeys,
            inputSpeed_callback=lambda: self.inputSpeed.get(), # Pass the slider value as a callback
            minDetConf=self.minDetConf.get(),  # Pass detection confidence
            minTrackConf=self.minTrackConf.get()
          ) 
        
        self.handRecog.landpointClassifier.load_model(modelPath)  # Load the selected model
        with open(labelPath, encoding='utf-8-sig') as f:
            keypointClass_labels = csv.reader(f)
            self.handRecog.landpointClassLabels = [row[0] for row in keypointClass_labels]  # Update labels

        self.running = True
        self.startButton.config(state=tk.DISABLED)
        self.stopButton.config(state=tk.NORMAL)
        self.updateVideo()

    def stopRecognition(self):
        # Stop the hand gesture recognition
        self.running = False
        self.startButton.config(state=tk.NORMAL)
        self.stopButton.config(state=tk.DISABLED)
        if self.handRecog:
            self.handRecog.cap.release()
        self.videoLabel.config(image="")

    def updateVideo(self):
        def video_loop():
            while self.running and self.handRecog:
                frame = self.handRecog.processFrame()  # Get the processed frame             
                if frame is not None and isinstance(frame, np.ndarray):
                    if frame.dtype == np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
                    image = Image.fromarray(frame)  # Convert to PIL Image
                    photo = ImageTk.PhotoImage(image=image)  # Convert to ImageTk format
                    self.videoLabel.config(image=photo)
                    self.videoLabel.image = photo

        # Start the video loop in a separate thread
        videoThread = threading.Thread(target=video_loop, daemon=True)
        videoThread.start()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()