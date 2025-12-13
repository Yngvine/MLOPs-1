""" Dummy image classification model and utility functions.
"""
import numpy as np
import random
import os
import cv2
from typing import cast

# Global variable to hold the ONNX session
_ORT_SESSION = None

def _get_ort_session():
    """Lazy load the ONNX session."""
    global _ORT_SESSION
    if _ORT_SESSION is None:
        try:
            import onnxruntime as ort
            # Assuming the model is in the 'model' directory at the project root
            # mylib/classifier.py -> ../model/oxford_pets_mobilenetv2.onnx
            model_path = os.path.join(os.path.dirname(__file__), "..", "model", "oxford_pets_mobilenetv2.onnx")
            model_path = os.path.abspath(model_path)
            if os.path.exists(model_path):
                _ORT_SESSION = ort.InferenceSession(model_path)
            else:
                print(f"Warning: ONNX model not found at {model_path}")
        except ImportError:
            print("Warning: onnxruntime not installed.")
    return _ORT_SESSION

def predict_class(_img: np.ndarray, n: int) -> int:
    """Predict the class of an image using the trained ONNX model.
    
    Parameters
    ----------
    _img : np.ndarray
        Input image as a numpy array.
    n : int
        Number of classes (unused, kept for API compatibility).

    Returns
    -------
    int
        Predicted class label.
        
    Raises
    ------
    RuntimeError
        If the ONNX model is not found or inference fails.
    """
    session = _get_ort_session()
    
    if session is None:
        raise RuntimeError("ONNX model not found. Please train and serialize the model first.")

    try:
        # Preprocessing for MobileNetV2
        # 1. Resize to 224x224
        img = cv2.resize(_img, (224, 224))
        
        # 2. Convert BGR to RGB (OpenCV uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # 4. Transpose to CHW (Channels, Height, Width)
        img = img.transpose(2, 0, 1)
        
        # 5. Add batch dimension (1, C, H, W)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img})
        output_tensor = cast(np.ndarray, outputs[0])
        predicted_class = np.argmax(output_tensor)
        
        return int(predicted_class)
    except Exception as e:
        raise RuntimeError(f"Error during inference: {e}") from e

def crop_image(img: np.ndarray, size: tuple) -> np.ndarray:
    """Crop an image to the given size.
    
    Parameters
    ----------
    img : np.ndarray
        Input image as a numpy array.
    size : tuple
        Desired size (width, height).

    Returns
    -------
    np.ndarray
        Resized image.
    """
    img_shape = img.shape

    start_indxes = (img_shape[0] - size[1]) // 2, (img_shape[1] - size[0]) // 2
    end_indxes = start_indxes[0] + size[1], start_indxes[1] + size[0]

    return img[start_indxes[0]:end_indxes[0], start_indxes[1]:end_indxes[1]]

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize an image to have pixel values between 0 and 1.
    
    Parameters
    ----------
    img : np.ndarray
        Input image as a numpy array.

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    img_min = img.min()
    img_max = img.max()
    divisor = img_max - img_min
    # Replace zero divisor with one to avoid division by zero
    if divisor == 0:
        divisor = 1
    normalized_img = (img - img_min) / divisor
    return normalized_img