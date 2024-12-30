import cv2
import os
import shutil
import numpy as np
from skimage import io, feature, color
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import disk, dilation, opening, closing
import math
from matplotlib import pyplot as plt
import math
from matplotlib import pyplot as plt


def display_results(images, titles):
    # You can change these values as needed:
    fig_width = 12     # Figure width in inches
    fig_height = 8     # Figure height in inches
    ncols = 3          # Number of images per row

    # Calculate how many rows are needed
    nrows = math.ceil(len(images) / ncols)

    # Create the grid of subplots
    fig, axes = plt.subplots(nrows=nrows, 
                             ncols=ncols, 
                             figsize=(fig_width, fig_height))

    # If nrows==1 or ncols==1, axes may be 1D; make it 1D either way for easy iteration
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # already 1D
    elif ncols == 1:
        axes = axes  # already 1D
    else:
        # Flatten the 2D array of axes to 1D
        axes = axes.ravel()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title, fontsize=16)
        axes[i].axis('off')

    # Hide any unused subplots (if total images < nrows * ncols)
    for j in range(len(images), len(axes)):
        axes[j].set_visible(False)

    # Make layout tight so rows and columns are nicely spaced
    plt.tight_layout()
    plt.show()

def save_results(images, titles, save_path):
    # You can change these values as needed:
    fig_width = 12     # Figure width in inches
    fig_height = 8     # Figure height in inches
    ncols = 3          # Number of images per row

    # Calculate how many rows are needed
    nrows = math.ceil(len(images) / ncols)

    # Create the grid of subplots
    fig, axes = plt.subplots(nrows=nrows, 
                             ncols=ncols, 
                             figsize=(fig_width, fig_height))

    # If nrows==1 or ncols==1, axes may be 1D; make it 1D either way for easy iteration
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes  # already 1D
    elif ncols == 1:
        axes = axes  # already 1D
    else:
        # Flatten the 2D array of axes to 1D
        axes = axes.ravel()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title, fontsize=16)
        axes[i].axis('off')

    # Hide any unused subplots (if total images < nrows * ncols)
    for j in range(len(images), len(axes)):
        axes[j].set_visible(False)

    # Make layout tight so rows and columns are nicely spaced
    plt.tight_layout()
    plt.savefig(save_path)  # Save the figure to the specified path
    plt.close(fig)  # Close the figure to free up memory



def rug_pattern_morph(image_path, lower_bound_pattern, upper_bound_background, object_l, background_l, gradient_sigma):
    # Step 1: Load and preprocess the image
    
    """
    Morphological segmentation of patterns in rug images.

    Parameters:
    image_path (str): Path to the image file to be processed.
    lower_bound_pattern (int): Lower bound of pixel intensity for the pattern.
    upper_bound_background (int): Upper bound of pixel intensity for the background.
    object_l (int): Label for objects (patterns) in the image.
    background_l (int): Label for the background.
    gradient_sigma (float): Standard deviation for the gradient filter used in edge detection.

    Returns:
    im_color (ndarray): The original image in RGB format.
    im_inv (ndarray): The inverted image.
    im_gradient (ndarray): The gradient of the inverted image after edge detection.
    markers (ndarray): The markers for the Watershed algorithm.
    masked_image (ndarray): The original image with the pattern masked out.
    """
    im_color = io.imread(image_path)
    im_g = color.rgb2gray(im_color)
    im = img_as_ubyte(im_g)
    im_inv = np.invert(im)

    # Apply opening and then closing to morphological smoothing
    im_inv = opening(im_inv, disk(1))
    im_inv = closing(im_inv, disk(1))

    # Step 2: Compute gradient using edge detection
    im_gradient = feature.canny(im_inv, gradient_sigma)
    im_gradient = dilation(im_gradient, disk(2))  # Thicken edges

    # Step 3: Define markers for masking
    markers = np.zeros_like(im)
    markers[im_inv > lower_bound_pattern] = object_l  # Foreground (patterns)
    markers[im_inv < upper_bound_background] = background_l  # Background

    markers = cv2.medianBlur(markers, 3)

    # Step 4: Apply the inverted mask to the original image
    mask = np.logical_not(markers.astype(bool))  # Invert the mask
    masked_image = np.zeros_like(im_color)
    for i in range(3):  # Assuming RGB image
        masked_image[..., i] = im_color[..., i] * mask

    return im_color, im_inv, im_gradient, markers, masked_image

def wrinkle_finder(image_path, sigma, threshold, morph_kernel):
    # Step 1: Load and preprocess the image
    im_color = io.imread(image_path)
    im_gray = color.rgb2gray(im_color)  # Convert to grayscale
    im_norm = cv2.normalize(im_gray, None, 0, 255, cv2.NORM_MINMAX)  # Normalize for consistency
    im_norm = np.uint8(im_norm)

    # Step 2: Find gradient
    im_gradient = feature.canny(im_norm, sigma)
    im_gradient = dilation(im_gradient, disk(2))  # Thicken edges
    im_gradient = np.abs(im_gradient)
    im_gradient = (im_gradient / im_gradient.max() * 255).astype(np.uint8)

    # Step 3: Threshold the edges
    _, binary_mask = cv2.threshold(im_gradient, threshold, 255, cv2.THRESH_BINARY)

    # Step 4: Postprocess using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    binary_mask = opening(binary_mask, kernel)  # Remove small noise
    binary_mask = closing(binary_mask, kernel)  # Fill small gaps

    return im_color, im_gradient, binary_mask

def zipper_finder(image_path, grad_lower, grad_upper, median_blur_k):
    # Step 1: Load and preprocess the image
    original = io.imread(image_path)
    grayscale = color.rgb2gray(original)
    grayscale = (grayscale * 255).astype(np.uint8)  # Normalize to 0-255

    # Step 2: Edge Detection
    gradient = cv2.medianBlur(grayscale, median_blur_k)
    gradient = cv2.Canny(gradient, grad_lower, grad_upper, apertureSize=3)

    # Step 3: thickening
    thickened = gradient.copy()  # Start with the original gradient
    # Correct way to create a structuring element
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    for k in range(5):
        dilated = cv2.morphologyEx(thickened, cv2.MORPH_DILATE, struct_elem)  # Apply dilation
        thickened = cv2.bitwise_or(thickened, dilated)  # Union operation (logical OR)

    # Step 4: Extract RGB regions of the zipper
    # Ensure the binary mask is a single channel with values 0 or 255
    thickened_binary = cv2.threshold(thickened, 127, 255, cv2.THRESH_BINARY)[1]
    zipper_rgb = cv2.bitwise_and(original, original, mask=thickened_binary)

    return original, gradient, thickened, zipper_rgb

if os.path.exists("results"):
    shutil.rmtree("results")
os.makedirs("results")

# Test the outputs
image_path1 = "THE3_Images/1.png"
try:
    # Process the image
    im_color1, im_inv1, im_gradient1, markers1, masked_image1 = rug_pattern_morph(
        image_path1, 10, 150, 255, 0, 2
    )
    # Display results
    display_results(
        [im_color1, im_inv1, im_gradient1, markers1, masked_image1],
        ["Image", "Inverted Image", "Image Gradient", "Markers", "Masked Image"]
    )
    save_results(
        [im_color1, im_inv1, im_gradient1, markers1, masked_image1],
        ["Image", "Inverted Image", "Image Gradient", "Markers", "Masked Image"],
        "results/morph1.png"
    )
except FileNotFoundError:
    print(f"File not found: {image_path1}")

image_path2 = "THE3_Images/2.png"
try:
    # Process the image
    im_color2, im_inv2, im_gradient2, markers2, masked_image2 = rug_pattern_morph(
        image_path2, 40, 180, 255, 0, 2.5
    )
    # Display results
    display_results(
        [im_color2, im_inv2, im_gradient2, markers2, masked_image2],
        ["Image", "Inverted Image", "Image Gradient", "Markers", "Masked Image"]
    )
    save_results(
        [im_color2, im_inv2, im_gradient2, markers2, masked_image2],
        ["Image", "Inverted Image", "Image Gradient", "Markers", "Masked Image"],
        "results/morph2.png"
    )
except FileNotFoundError:
    print(f"File not found: {image_path2}")

image_path3 = "THE3_Images/3.png"
try:
    # Process the image to detect wrinkles
    original, edges, binary_mask = wrinkle_finder(
        image_path3, 1.33, 20, 3
    )
    # Display the results
    display_results(
        [original, edges, binary_mask],
        ["Original Image", "Gradient", "Wrinkle Mask"]
    )
    save_results(
        [original, edges, binary_mask],
        ["Original Image", "Gradient", "Wrinkle Mask"],
        "results/morph3.png"
    )
except FileNotFoundError:
    print(f"File not found: {image_path3}")

image_path4 = "THE3_Images/4.png"
try:
    # Process the image to detect wrinkles
    original, edges, binary_mask = wrinkle_finder(
        image_path4, 2, 20, 3
    )
    # Display the results
    display_results(
        [original, edges, binary_mask],
        ["Original Image", "Gradient", "Wrinkle Mask"]
    )
    save_results(
        [original, edges, binary_mask],
        ["Original Image", "Gradient", "Wrinkle Mask"],
        "results/morph4.png"
    )
except FileNotFoundError:
    print(f"File not found: {image_path4}")

image_path5 = "THE3_Images/5.png"  # Replace with your zipper image path
try:
    # Detect zippers in the image
    original, edges, line_image, rgb = zipper_finder(image_path5, 50, 150, 3)
    # Display the results
    display_results(
        [original, edges, line_image, rgb],
        ["Original Image", "Canny Edges", "Detected Lines (Zipper)", "RGB"]
    )
    save_results(
        [original, edges, line_image, rgb],
        ["Original Image", "Canny Edges", "Detected Lines (Zipper)", "RGB"],
        "results/morph5.png"
    )
except FileNotFoundError:
    print(f"File not found: {image_path5}")

image_path6 = "THE3_Images/6.png"  # Replace with your zipper image path
try:
    # Detect zippers in the image
    original, edges, line_image, rgb = zipper_finder(image_path6, 100, 200, 7)
    # Display the results
    display_results(
        [original, edges, line_image, rgb],
        ["Original Image", "Canny Edges", "Detected Lines (Zipper)", "RGB"]
    )
    save_results(
        [original, edges, line_image, rgb],
        ["Original Image", "Canny Edges", "Detected Lines (Zipper)", "RGB"],
        "results/morph6.png"
    )
except FileNotFoundError:
    print(f"File not found: {image_path6}")
