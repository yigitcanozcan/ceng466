import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from common_operations import gaussian_smoothing, contrast_stretching


def preprocess_image(image_path, kernel_size):
    # Step 1: Load the RGB image
    original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Step 2: Apply Gaussian smoothing
    smoothed = gaussian_smoothing(original, kernel_size)

    # Step 3: Apply contrast stretching to each channel separately
    stretched = np.zeros_like(smoothed)
    for i in range(3):  # Process each channel (R, G, B) independently
        stretched[..., i] = contrast_stretching(smoothed[..., i])

    return original, stretched


def apply_kmeans(image, n_clusters=3):
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
    # Create a binary mask for the target cluster
    mask = (labels == target_cluster).astype(np.uint8) * 255

    # Convert mask to 3 channels to match the RGB image
    mask_3channel = cv2.merge([mask] * 3)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original, mask_3channel)
    return masked_image

def mask_segment_enhanced(original, labels, target_clusters):
    # Initialize a blank mask
    combined_mask = np.zeros_like(labels, dtype=np.uint8)

    # Combine masks for all target clusters
    for cluster in target_clusters:
        combined_mask[labels == cluster] = 255

    # Convert combined mask to 3 channels to match the original image
    mask_3channel = cv2.merge([combined_mask] * 3)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original, mask_3channel)

    return masked_image


def display_results(titles, images):
    plt.figure(figsize=(20, 8))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, len(images), i + 1)
        plt.title(title)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Test the outputs
image_path1 = "THE3_Images/1.png"

# Preprocessing
original, preprocessed = preprocess_image(image_path1, 7)

# Apply KMeans Clustering
segmented_image, labels = apply_kmeans(preprocessed, n_clusters=2)

# Mask the Segment (focus on a specific cluster, e.g., cluster 0)
masked_segment = mask_segment(original, labels, target_cluster=1)

# Display Results
titles = ["Original Image", "Preprocessed Image", "Segmented Image", "Masked Segment"]
images = [original, preprocessed, segmented_image, masked_segment]
display_results(titles, images)

image_path2 = "THE3_Images/2.png"

# Preprocessing
original, preprocessed = preprocess_image(image_path2, 3)

# Apply KMeans Clustering
segmented_image, labels = apply_kmeans(preprocessed, n_clusters=2)

# Mask the Segment (focus on a specific cluster, e.g., cluster 0)
masked_segment = mask_segment(original, labels, target_cluster=1)

# Display Results
titles = ["Original Image", "Preprocessed Image", "Segmented Image", "Masked Segment"]
images = [original, preprocessed, segmented_image, masked_segment]
display_results(titles, images)


image_path3 = "THE3_Images/3.png"

# Preprocessing
original, preprocessed = preprocess_image(image_path3, 3)

# Apply KMeans Clustering
segmented_image, labels = apply_kmeans(preprocessed, n_clusters=7)

# Clustering
target_clusters = [2, 4, 5, 6]
masked_segment = mask_segment_enhanced(original, labels, target_clusters)

# Display Results
titles = ["Original Image", "Preprocessed Image", "Segmented Image", "Masked Segment"]
images = [original, preprocessed, segmented_image, masked_segment]
display_results(titles, images)

image_path4 = "THE3_Images/4.png"

# Preprocessing
original, preprocessed = preprocess_image(image_path4, 9)

# Apply KMeans Clustering
segmented_image, labels = apply_kmeans(preprocessed, n_clusters=5)

# Mask the Segment
target_clusters = [1,2]
masked_segment = mask_segment_enhanced(original, labels, target_clusters)

# Display Results
titles = ["Original Image", "Preprocessed Image", "Segmented Image", "Masked Segment"]
images = [original, preprocessed, segmented_image, masked_segment]
display_results(titles, images)

image_path5 = "THE3_Images/5.png"

# Preprocessing
original, preprocessed = preprocess_image(image_path5, 5)

# Apply KMeans Clustering
segmented_image, labels = apply_kmeans(preprocessed, n_clusters=3)

# Mask the Segment (focus on a specific cluster, e.g., cluster 0)
masked_segment = mask_segment(original, labels, target_cluster=0)

# Display Results
titles = ["Original Image", "Preprocessed Image", "Segmented Image", "Masked Segment"]
images = [original, preprocessed, segmented_image, masked_segment]
display_results(titles, images)

image_path6 = "THE3_Images/6.png"
target_clusters = [2,3]

# Preprocessing
original, preprocessed = preprocess_image(image_path6, 13)

# Apply KMeans Clustering
segmented_image, labels = apply_kmeans(preprocessed, n_clusters=5)

# Mask the Segment (focus on a specific cluster, e.g., cluster 0)
masked_segment = mask_segment_enhanced(original, labels, target_clusters)

# Display Results
titles = ["Original Image", "Preprocessed Image", "Segmented Image", "Masked Segment"]
images = [original, preprocessed, segmented_image, masked_segment]
display_results(titles, images)

# 6.png is the hardest one, because of the shadows :(

