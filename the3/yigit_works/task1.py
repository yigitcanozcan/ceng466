import cv2
import numpy as np
from skimage import io, feature, color
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import disk, dilation, opening, closing


def display_results(images, titles):
    plt.figure(figsize=(20, 15))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title, fontsize=16)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def rug_pattern_morph(image_path, lower_bound_pattern, upper_bound_background, object_l, background_l, gradient_sigma):
    # Step 1: Load and preprocess the image
    im_color = io.imread(image_path)
    im_g = color.rgb2gray(im_color)
    im = img_as_ubyte(im_g)
    im_inv = np.invert(im)

    # Apply opening and then closing to morphological smoothing
    im_inv = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, disk(1))
    im_inv = cv2.morphologyEx(im_inv, cv2.MORPH_CLOSE, disk(1))

    # Step 2: Compute gradient using edge detection
    im_gradient = feature.canny(im_inv, gradient_sigma)
    im_gradient = dilation(im_gradient, disk(2))  # Thicken edges

    # Step 3: Define markers for Watershed
    markers = np.zeros_like(im)
    markers[im_inv > lower_bound_pattern] = object_l # Foreground (patterns)
    markers[im_inv < upper_bound_background] = background_l  # Background

    markers = cv2.medianBlur(markers, 3)

    label_img = label(markers)

    # Step 4: Apply the Watershed algorithm
    labels = watershed(im_gradient, markers=label_img)


    return im_color, im_inv, im_gradient, markers, labels


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



# Test the outputs
image_path1 = "THE3_Images/1.png"
try:
    # Process the image
    im_color1, im_inv1, im_gradient1, markers1, labels1 = rug_pattern_morph(image_path1, 10, 150, 7, 3, 2)
    # Display results
    display_results(
        [im_color1, im_inv1, im_gradient1, markers1.astype('int'), labels1],
        ["Image", "Inverted Image", "Image Gradient", "Markers", "Watershed Segmentation"]
    )
except FileNotFoundError:
    print(f"File not found: {image_path1}")

image_path2 = "THE3_Images/2.png"
try:
    # Process the image
    im_color2, im_inv2, im_gradient2, markers2, labels2 = rug_pattern_morph(image_path2, 40, 180, 7, 3, 2.5)
    # Display results
    display_results(
        [im_color2, im_inv2, im_gradient2, markers2.astype('int'), labels2],
        ["Image", "Inverted Image", "Image Gradient", "Markers", "Watershed Segmentation"]
    )
except FileNotFoundError:
    print(f"File not found: {image_path2}")
    
image_path3 = "THE3_Images/3.png"

try:
    # Process the image to detect wrinkles
    original, edges, binary_mask = wrinkle_finder(
        image_path3, 1.33, 20, 3)

    # Display the results
    display_results(
        [original, edges, binary_mask],
        ["Original Image", "Gradient", "Wrinkle Mask"]
    )
except FileNotFoundError:
    print(f"File not found: {image_path3}")

image_path4 = "THE3_Images/4.png"

try:
    # Process the image to detect wrinkles
    original, edges, binary_mask = wrinkle_finder(
        image_path4, 2, 20, 3)

    # Display the results
    display_results(
        [original, edges, binary_mask],
        ["Original Image", "Gradient", "Wrinkle Mask"]
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
except FileNotFoundError:
    print(f"File not found: {image_path6}")



