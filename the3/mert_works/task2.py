import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from common_operations import gaussian_smoothing, contrast_stretching

def preprocess_image(image_path):
    """
    Preprocess the image by loading, smoothing, and contrast stretching.
    
    Parameters:
        image_path (str): Path to the input image.
    
    Returns:
        original (numpy array): Original RGB image.
        preprocessed (numpy array): Preprocessed RGB image after smoothing and stretching.
    """
    # Step 1: Load the RGB image
    original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    # Step 2: Apply Gaussian smoothing
    smoothed = gaussian_smoothing(original, kernel_size=5)
    
    # Step 3: Apply contrast stretching to each channel separately
    stretched = np.zeros_like(smoothed)
    for i in range(3):  # Process each channel (R, G, B) independently
        stretched[..., i] = contrast_stretching(smoothed[..., i])
    
    return original, stretched

def apply_kmeans(image, n_clusters=3):
    """
    Apply KMeans clustering to segment the image based on RGB features.
    
    Parameters:
        image (numpy array): Input RGB image.
        n_clusters (int): Number of clusters for KMeans.
    
    Returns:
        segmented_image (numpy array): Segmented image.
        labels (numpy array): Labels of clusters for each pixel.
    """
    # Reshape the image into a 2D array of pixels with 3 RGB features
    pixels = image.reshape((-1, 3))
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    
    # Get the cluster labels for each pixel
    labels = kmeans.labels_
    
    # Map the cluster centers back to image
    cluster_centers = np.uint8(kmeans.cluster_centers_)
    segmented_pixels = cluster_centers[labels]
    
    # Reshape back to the original image dimensions
    segmented_image = segmented_pixels.reshape(image.shape)
    
    return segmented_image, labels.reshape(image.shape[:2])

def mask_segment(original, labels, target_cluster):
    """
    Create a mask for the target cluster and apply it to the original image.
    
    Parameters:
        original (numpy array): Original RGB image.
        labels (numpy array): Labels array from KMeans clustering.
        target_cluster (int): The cluster of interest to extract.
    
    Returns:
        masked_image (numpy array): Image with only the target cluster visible.
    """
    # Create a binary mask for the target cluster
    mask = (labels == target_cluster).astype(np.uint8) * 255
    
    # Convert mask to 3 channels to match the RGB image
    mask_3channel = cv2.merge([mask] * 3)
    
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original, mask_3channel)
    return masked_image

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
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main Script
image_path = "THE3_IMAGES/3.png"  # Replace with your image path

# Preprocessing
original, preprocessed = preprocess_image(image_path)

# Apply KMeans Clustering
segmented_image, labels = apply_kmeans(preprocessed, n_clusters=3)

# Mask the Segment (focus on a specific cluster, e.g., cluster 0)
masked_segment = mask_segment(original, labels, target_cluster=0)

# Display Results
titles = ["Original Image", "Preprocessed Image", "Segmented Image", "Masked Segment"]
images = [original, preprocessed, segmented_image, masked_segment]
display_results(titles, images)
