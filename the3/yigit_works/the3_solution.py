

# CENG466 THE3
# Mert Uludoğan 2380996
# Yiğitcan Özcan 2521847


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
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from common_operations import gaussian_smoothing, contrast_stretching
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from skimage.filters.rank import threshold
from sklearn.cluster import KMeans

from common_operations import gaussian_smoothing

class Task1():

    def __init__(self):
        pass

    def run(self):
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

        # if os.path.exists("results"):
        #     shutil.rmtree("results")
        # os.makedirs("results")

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
            # save_results(
            #     [im_color1, im_inv1, im_gradient1, markers1, masked_image1],
            #     ["Image", "Inverted Image", "Image Gradient", "Markers", "Masked Image"],
            #     "results/morph1.png"
            # )
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
            # save_results(
            #     [im_color2, im_inv2, im_gradient2, markers2, masked_image2],
            #     ["Image", "Inverted Image", "Image Gradient", "Markers", "Masked Image"],
            #     "results/morph2.png"
            # )
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
            # save_results(
            #     [original, edges, binary_mask],
            #     ["Original Image", "Gradient", "Wrinkle Mask"],
            #     "results/morph3.png"
            # )
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
            # save_results(
            #     [original, edges, binary_mask],
            #     ["Original Image", "Gradient", "Wrinkle Mask"],
            #     "results/morph4.png"
            # )
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
            # save_results(
            #     [original, edges, line_image, rgb],
            #     ["Original Image", "Canny Edges", "Detected Lines (Zipper)", "RGB"],
            #     "results/morph5.png"
            # )
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
            # save_results(
            #     [original, edges, line_image, rgb],
            #     ["Original Image", "Canny Edges", "Detected Lines (Zipper)", "RGB"],
            #     "results/morph6.png"
            # )
        except FileNotFoundError:
            print(f"File not found: {image_path6}")

class Task2():

    def __init__(self):
        pass

    def run(self):
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

class Task3():

    def __init__(self):
        pass

    def run(self):
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



if __name__ == "__main__":

    task1 = Task1()
    task1.run()

    task2 = Task2()
    task2.run()

    task3 = Task3()
    task3.run()








