import numpy as np
import tensorflow as tf


class LandPointClassifier(object):
    def __init__(
        self,
        modelPath='utils/model/landPointClassifier.tflite', numThreads=1, 
        ):
        # Initialize the TensorFlow Lite interpreter with the specified model and number of threads
        self.interpreter = tf.lite.Interpreter(model_path=modelPath, num_threads=numThreads)

        self.interpreter.allocate_tensors() # Allocate memory for the model's tensors
        self.inputDetails = self.interpreter.get_input_details() # Get details about the model's input tensor
        self.outputDetails = self.interpreter.get_output_details() # Get details about the model's output tensor

    def load_model(self, model_path):
        """
        Dynamically load a new TFLite model.
        This allows switching models without reinitializing the class.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inputDetails = self.interpreter.get_input_details()
        self.outputDetails = self.interpreter.get_output_details()
    

    def __call__(
        self,
        landmark_list,
    ):
        """
        Perform inference on the given landmark list and return the classification result.
        """
        inputDetTensorIndex = self.inputDetails[0]['index']
        self.interpreter.set_tensor(
            inputDetTensorIndex,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke() # Run inference on the input data

        outputDetTensorIndex = self.outputDetails[0]['index']

        # Retrieve the output tensor (classification results)
        result = self.interpreter.get_tensor(outputDetTensorIndex)
        
        # Find the index of the highest probability in the output (the predicted class)
        resultIndex = np.argmax(np.squeeze(result))

        return resultIndex
