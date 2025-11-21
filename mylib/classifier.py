""" Dummy image classification model and utility functions.
"""
import numpy as np
import random

def predict_class(img: np.ndarray, n: int) -> int:
    """Predict the class of an image using a dummy model.   
    
    Parameters
    ----------
    img : np.ndarray
        Input image as a numpy array.
    n : int
        Number of classes.

    Returns
    -------
    int
        Predicted class label.
    """
    # Dummy implementation: always return 0
    return random.randint(0, n-1)

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
    normalized_img = (img - img_min) / (img_max - img_min)
    return normalized_img