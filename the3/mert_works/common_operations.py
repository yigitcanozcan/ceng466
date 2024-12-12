
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_smoothing(image, kernel_size):
    """
    Apply Gaussian smoothing to reduce noise.
    
    Parameters:
        image (numpy array): Input grayscale image.
        kernel_size (int): Size of the Gaussian kernel.
    
    Returns:
        smoothed (numpy array): Smoothed image.
    """
    smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return smoothed

def contrast_stretching(image):
    """
    Perform contrast stretching to enhance the image.
    
    Parameters:
        image (numpy array): Input grayscale image.
    
    Returns:
        stretched (numpy array): Contrast-enhanced image.
    """
    # Compute minimum and maximum pixel values
    min_val, max_val = np.min(image), np.max(image)
    # Stretch the contrast
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched
