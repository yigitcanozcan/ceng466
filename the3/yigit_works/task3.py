import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from skimage.filters.rank import threshold
from sklearn.cluster import KMeans

from common_operations import gaussian_smoothing


def compute_lbp(image, radius=1, n_points=8):
    return local_binary_pattern(image, P=n_points, R=radius, method="default")


def apply_threshold_to_lbp(lbp_image, lower, upper):
    # Initialize a blank output image
    thresholded_image = np.zeros_like(lbp_image)

    # Keep only values within the specified range (grayish values)
    mask = (lbp_image >= lower) & (lbp_image <= upper)
    thresholded_image[mask] = lbp_image[mask]

    return thresholded_image

def apply_kmeans(image, n_clusters=3):
    # Step 3: Flatten the LBP image into a feature vector
    h, w = image.shape
    lbp_flat = image.reshape(-1, 1)

    # Step 4: Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(lbp_flat)

    # Step 5: Reshape the labels to match the original image
    clustered_image = labels.reshape(h, w)

    return clustered_image, labels.reshape(h,w)

def mask_segment_enhanced_lbp(original, labels, target_clusters):
    # Step 1: Convert original to uint8 if needed
    if original.dtype != np.uint8:
        original_uint8 = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        original_uint8 = original

    # Step 2: Initialize a blank mask
    mask = np.zeros_like(labels, dtype=np.uint8)

    # Step 3: Combine masks for all target clusters
    for cluster in target_clusters:
        mask[labels == cluster] = 255

    # Step 4: Reshape or resize the mask to match the original image dimensions
    if mask.shape != original.shape:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_3channel = cv2.merge([mask] * 3)

    # Step 5: Apply the mask to the original grayscale image
    masked_image = cv2.bitwise_and(original_uint8, mask_3channel)

    return masked_image



def display_results(titles, images, cmap='gray'):
    plt.figure(figsize=(20, 8))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, len(images), i + 1)
        plt.title(title)
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def remove_small_objects_morph(lbp_image, kernel_size=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_image = cv2.morphologyEx(lbp_image, cv2.MORPH_DILATE, kernel, iterations=iterations)

    return dilated_image


def adjust_pixel_intensity(image, lower, upper, darken_factor=0.2, lighten_factor=5):
    # Ensure the input image is grayscale
    image = image.astype(np.float32)  # Work with float for scaling

    # Create masks for darkening and lightening
    darken_mask = image < lower
    lighten_mask = image > upper

    # Adjust the intensity
    adjusted_image = image.copy()
    adjusted_image[darken_mask] *= darken_factor
    adjusted_image[lighten_mask] *= lighten_factor

    # Clip the values to be in the valid range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image



# Test the outputs

image_path1 = "THE3_Images/1.png"  # Replace with your image path

# Load the grayscale image
original = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
smoothed = gaussian_smoothing(original, 49)
original_bgr = cv2.imread(image_path1, cv2.IMREAD_COLOR)
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
smoothed_color = gaussian_smoothing(original_rgb, 49)

# Compute LBP
lbp_image = compute_lbp(smoothed, radius=1, n_points=8)

clustered, labels = apply_kmeans(lbp_image, n_clusters=5)
target_clusters = [1,2,4]
masked_segment = mask_segment_enhanced_lbp(original_rgb, labels, target_clusters)

# Display Results
titles = ["Smoothed Image", "LBP Image", "Clustered", "Masked segment"]
images = [smoothed, lbp_image, clustered, masked_segment]
display_results(titles, images)


image_path2 = "THE3_Images/2.png"  # Replace with your image path

# Load the grayscale image
original = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
smoothed = gaussian_smoothing(original, 5)
original_bgr = cv2.imread(image_path2, cv2.IMREAD_COLOR)
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
smoothed_color = gaussian_smoothing(original_rgb, 5)

# Compute LBP
lbp_image = compute_lbp(smoothed, radius=5, n_points=8)

clustered, labels = apply_kmeans(lbp_image, n_clusters=8)
target_clusters = [2,7]
masked_segment = mask_segment_enhanced_lbp(original_rgb, labels, target_clusters)

# Display Results
titles = ["Smoothed Image", "LBP Image", "Clustered", "Masked segment"]
images = [smoothed, lbp_image, clustered, masked_segment]
display_results(titles, images)

image_path3 = "THE3_Images/3.png"  # Replace with your image path

# Load the grayscale image
original = cv2.imread(image_path3, cv2.IMREAD_GRAYSCALE)
smoothed = gaussian_smoothing(original, 5)
original_bgr = cv2.imread(image_path3, cv2.IMREAD_COLOR)
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
smoothed_color = gaussian_smoothing(original_rgb, 5)

# Compute LBP
lbp_image = compute_lbp(smoothed, radius=5, n_points=8)

clustered, labels = apply_kmeans(lbp_image, n_clusters=3)
target_clusters = [0]
masked_segment = mask_segment_enhanced_lbp(original_rgb, labels, target_clusters)

# Display Results
titles = ["Smoothed Image", "LBP Image", "Clustered", "Masked segment"]
images = [smoothed, lbp_image, clustered, masked_segment]
display_results(titles, images)

image_path4 = "THE3_Images/4.png"  # Replace with your image path

# Load the grayscale image
original = cv2.imread(image_path4, cv2.IMREAD_GRAYSCALE)
smoothed = gaussian_smoothing(original, 17)
original_bgr = cv2.imread(image_path4, cv2.IMREAD_COLOR)
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
smoothed_color = gaussian_smoothing(original_rgb, 17)

# Compute LBP
lbp_image = compute_lbp(smoothed, radius=1, n_points=8)
lbp_image = remove_small_objects_morph(lbp_image)

clustered, labels = apply_kmeans(lbp_image, n_clusters=5)
target_clusters = [1,2,3,4]
masked_segment = mask_segment_enhanced_lbp(original_rgb, labels, target_clusters)


# Display Results
titles = ["Smoothed Image", "LBP Image (dilated)", "Clustered", "Masked segment"]
images = [smoothed, lbp_image, clustered, masked_segment]
display_results(titles, images)


image_path5 = "THE3_Images/5.png"  # Replace with your image path

# Load the grayscale image
original = cv2.imread(image_path5, cv2.IMREAD_GRAYSCALE)

# Bright pixels brighter, dark pixels darker
original = adjust_pixel_intensity(original, 120, 120)

smoothed = gaussian_smoothing(original, 17)
original_bgr = cv2.imread(image_path5, cv2.IMREAD_COLOR)
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
smoothed_color = gaussian_smoothing(original_rgb, 17)

# Compute LBP
lbp_image = compute_lbp(smoothed, radius=3, n_points=8)
lbp_image = remove_small_objects_morph(lbp_image, kernel_size=3)

clustered, labels = apply_kmeans(lbp_image, n_clusters=5)
target_clusters = [0,2,3,4]
masked_segment = mask_segment_enhanced_lbp(original_rgb, labels, target_clusters)


# Display Results
titles = ["Smoothed Image", "LBP Image (dilated)", "Clustered", "Masked segment"]
images = [smoothed, lbp_image, clustered, masked_segment]
display_results(titles, images)


image_path6 = "THE3_Images/6.png"  # Replace with your image path

# Load the grayscale image
original = cv2.imread(image_path6, cv2.IMREAD_GRAYSCALE)
smoothed = gaussian_smoothing(original, 17)
original_bgr = cv2.imread(image_path6, cv2.IMREAD_COLOR)
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
smoothed_color = gaussian_smoothing(original_rgb, 17)

# Bright pixels brighter, dark pixels darker
smoothed = adjust_pixel_intensity(smoothed, 0, 110, darken_factor=0.1, lighten_factor=150)

# Compute LBP
lbp_image = compute_lbp(smoothed, radius=3, n_points=8)
lbp_image = remove_small_objects_morph(lbp_image, kernel_size=3)

clustered, labels = apply_kmeans(lbp_image, n_clusters=5)
target_clusters = [0,1,2,3]
masked_segment = mask_segment_enhanced_lbp(original_rgb, labels, target_clusters)


# Display Results
titles = ["Smoothed (+ pixel adjusted) Image", "LBP Image (dilated)", "Clustered", "Masked segment"]
images = [smoothed, lbp_image, clustered, masked_segment]
display_results(titles, images)