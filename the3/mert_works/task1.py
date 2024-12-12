import cv2
import numpy as np
import matplotlib.pyplot as plt

from common_operations import gaussian_smoothing, contrast_stretching


def preprocess_image(image_path):
    """
    Preprocess the image by loading, smoothing, and contrast stretching.
    
    Parameters:
        image_path (str): Path to the input image.
        gaussian_kernel (int): Kernel size for Gaussian smoothing.
    
    Returns:
        original (numpy array): Original grayscale image.
        preprocessed (numpy array): Preprocessed image after smoothing and stretching.
    """
    # Step 1: Load the grayscale image
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 2: Apply Gaussian smoothing
    smoothed = gaussian_smoothing(original, kernel_size=5)
    
    # Adaptive Thresholding
    # Step 3: Apply contrast stretching
    stretched = contrast_stretching(smoothed)
    
    thresholded = cv2.adaptiveThreshold(
        stretched, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 7
    )

    return original, thresholded

def postprocess_image(binary_mask, kernel_size=5):
    """
    Refine the binary mask using morphological operations (opening and closing).
    
    Parameters:
        binary_mask (numpy array): Binary image to be processed.
        kernel_size (int): Size of the structuring element.
    
    Returns:
        refined_mask (numpy array): Binary mask after morphological refinement.
    """
    # Create a structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply opening to remove noise
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply closing to fill small gaps in the patterns
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed

def extract_patterns(original, mask):
    """
    Extract patterns from the original image using the refined binary mask.
    
    Parameters:
        original (numpy array): Original grayscale image.
        mask (numpy array): Refined binary mask.
    
    Returns:
        extracted (numpy array): Image with extracted patterns.
    """
    return cv2.bitwise_and(original, original, mask=mask)

def display_results(titles, images):
    """
    Display multiple images in a single plot for comparison.
    
    Parameters:
        titles (list): List of titles for the images.
        images (list): List of image arrays to display.
    """
    plt.figure(figsize=(20, 8))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, len(images), i + 1)
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main Script
image_path = "THE3_IMAGES/2.png"  # Replace with your image path

# Preprocessing
original, preprocessed = preprocess_image(image_path)

# Postprocessing
refined_mask = postprocess_image(preprocessed)

# Extract Patterns
extracted_patterns = extract_patterns(original, refined_mask)

# Display Results
titles = ["Original Image", "Preprocessed Image", "Refined Mask", "Extracted Patterns"]
images = [original, preprocessed, refined_mask, extracted_patterns]
display_results(titles, images)
