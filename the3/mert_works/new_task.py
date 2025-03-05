


import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.morphology import remove_small_objects, remove_small_holes
from enum import Enum

import os
import shutil

import matplotlib.pyplot as plt
######################
# Parameter Constants
######################

IMG_FOLDER = "./THE3_IMAGES"


class SEGMENTATION_METHOD(Enum):
    GRAYSCALE_MORPHOLOGY = "morph"
    KMEANS_RGB = "km_rgb"
    KMEANS_LBP = "km_lbp"

class Pipeline:
    # General parameters
    # Morphological Parameters
    SEG_METHOD: SEGMENTATION_METHOD

    MORPH_KERNEL_SIZE = 5
    MORPH_ITERATIONS = 2
    MORPH_ELEMENT = cv2.MORPH_ELLIPSE  # or MORPH_RECT, MORPH_CROSS

    # KMeans Parameters for RGB
    KMEANS_CLUSTERS = 2
    KMEANS_INIT = 'k-means++'
    KMEANS_RANDOM_STATE = 42

    # KMeans Parameters for LBP
    KMEANS_LBP_RADIUS = 1
    KMEANS_LBP_POINTS = 8 * KMEANS_LBP_RADIUS
    KMEANS_LBP_METHOD = 'uniform'

    # Preprocessing Parameters
    BLUR_KERNEL_SIZE = (3, 3)  # for Gaussian blur
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)

    # Postprocessing Parameters
    MIN_OBJECT_SIZE = 50       # minimum object size in pixels to keep
    FILL_HOLES_SIZE = 50       # remove small holes up to this size

    orig_img = None
    save_folder = None
    log = None
    postfix = None

    def __init__(
            self,
            seg_method,
            morph_kernel_size=5,
            morph_iterations=2,
            morph_element=cv2.MORPH_ELLIPSE,
            kmeans_clusters=2,
            kmeans_init='k-means++',
            kmeans_random_state=42,
            kmeans_lbp_radius=1,
            kmeans_lbp_points=8 * 1,
            kmeans_lbp_method='uniform',
            blur_kernel_size=(3, 3),
            clahe_clip_limit=2.0,
            clahe_tile_size=(8, 8),
            min_object_size=50,
            fill_holes_size=50
    ):
        self.SEG_METHOD = seg_method
        self.MORPH_KERNEL_SIZE = morph_kernel_size
        self.MORPH_ITERATIONS = morph_iterations
        self.MORPH_ELEMENT = morph_element
        self.KMEANS_CLUSTERS = kmeans_clusters
        self.KMEANS_INIT = kmeans_init
        self.KMEANS_RANDOM_STATE = kmeans_random_state
        self.KMEANS_LBP_RADIUS = kmeans_lbp_radius
        self.KMEANS_LBP_POINTS = kmeans_lbp_points
        self.KMEANS_LBP_METHOD = kmeans_lbp_method
        self.BLUR_KERNEL_SIZE = blur_kernel_size
        self.CLAHE_CLIP_LIMIT = clahe_clip_limit
        self.CLAHE_TILE_SIZE = clahe_tile_size
        self.MIN_OBJECT_SIZE = min_object_size
        self.FILL_HOLES_SIZE = fill_holes_size

    @staticmethod
    def load_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not read image at {path}")
        return img

    def save_result(self, img):
        # Create a figure with a 2x2 grid:
        # Top row for text (left) and empty space (right),
        # Bottom row for original image (left) and segmented image (right).
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        
        # Top-left subplot for the log text
        axes[0,0].axis('off')
        axes[0,0].text(0.5, 0.5, self.log,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12,
                    transform=axes[0,0].transAxes)
        
        # Top-right subplot is unused, turn it off
        axes[0,1].axis('off')
        
        # Bottom-left subplot for the original image
        # Convert BGR (OpenCV format) to RGB for matplotlib display if needed
        orig_rgb = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB)
        axes[1,0].imshow(orig_rgb)
        axes[1,0].set_title("Original Image")
        axes[1,0].axis('off')
        
        # Bottom-right subplot for the segmented image
        axes[1,1].imshow(img, cmap='gray')
        axes[1,1].set_title("Segmented Image")
        axes[1,1].axis('off')
        
        # Adjust spacing
        plt.tight_layout()
        
        # Save the figure to a file
        seg_folder = os.path.join(self.save_folder, self.SEG_METHOD.value)
        os.makedirs(seg_folder, exist_ok=True)
        save_path = os.path.join(seg_folder, self.postfix + ".png")
        plt.savefig(save_path)
        plt.close(fig)


    def preprocessing(self, img, apply_blur=True, apply_clahe=True):
        self.postfix = f"_blur_{self.BLUR_KERNEL_SIZE}_clahe_{self.CLAHE_CLIP_LIMIT}_{self.CLAHE_TILE_SIZE}" + self.postfix
        # Convert to grayscale for certain steps (like CLAHE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        if apply_blur:
            gray = cv2.GaussianBlur(gray, self.BLUR_KERNEL_SIZE, 0)

        # Apply CLAHE for contrast enhancement
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=self.CLAHE_TILE_SIZE)
            gray = clahe.apply(gray)

        return gray

    def postprocessing(self, binary_mask):
        # binary_mask assumed to be a boolean mask or 0/255 image
        # Convert to boolean if needed
        if binary_mask.dtype != bool:
            binary_mask = binary_mask > 0

        # Remove small objects
        binary_mask = remove_small_objects(binary_mask, self.MIN_OBJECT_SIZE)

        # Fill small holes
        binary_mask = remove_small_holes(binary_mask, self.FILL_HOLES_SIZE)

        # Convert back to uint8
        binary_mask = (binary_mask * 255).astype(np.uint8)
        return binary_mask

    def grayscale_morphology_segmentation(self, gray):
        self.postfix = f"kernel_{self.MORPH_KERNEL_SIZE}_iter_{self.MORPH_ITERATIONS}_element_{self.MORPH_ELEMENT}" + self.postfix
        # Example pipeline:
        # 1. Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # 2. Morphological operations
        kernel = cv2.getStructuringElement(self.MORPH_ELEMENT, (self.MORPH_KERNEL_SIZE, self.MORPH_KERNEL_SIZE))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=self.MORPH_ITERATIONS)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=self.MORPH_ITERATIONS)

        return closed

    def kmeans_segmentation_rgb(self, img):
        # Reshape image to list of RGB pixels
        self.postfix = f"clusters_{self.KMEANS_CLUSTERS}_init_{self.KMEANS_INIT}" + self.postfix
        pixel_values = img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # Apply KMeans
        kmeans = KMeans(n_clusters=self.KMEANS_CLUSTERS, init=self.KMEANS_INIT, random_state=self.KMEANS_RANDOM_STATE)
        labels = kmeans.fit_predict(pixel_values)

        # Reshape back to image dimension
        segmented_image = labels.reshape((img.shape[0], img.shape[1]))

        return segmented_image

    def kmeans_segmentation_with_lbp(self, img):
        self.postfix = f"clusters_{self.KMEANS_LBP_RADIUS}_init_{self.KMEANS_LBP_METHOD}" + self.postfix
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute LBP for each pixel
        lbp = local_binary_pattern(gray, self.KMEANS_LBP_POINTS, self.KMEANS_LBP_RADIUS, method=self.KMEANS_LBP_METHOD)

        # LBP is a 2D array, we can directly use it as feature
        features = lbp.reshape(-1, 1).astype(np.float32)

        # Apply KMeans
        kmeans = KMeans(n_clusters=self.KMEANS_CLUSTERS, init=self.KMEANS_INIT, random_state=self.KMEANS_RANDOM_STATE)
        labels = kmeans.fit_predict(features)

        # Reshape to image dimensions
        segmented_image = labels.reshape((img.shape[0], img.shape[1]))

        return segmented_image

    def run_segmentation_pipeline(self, img):
        # Preprocessing
        if self.SEG_METHOD == SEGMENTATION_METHOD.GRAYSCALE_MORPHOLOGY:
            prep_gray = self.preprocessing(img, apply_blur=True, apply_clahe=True)
            seg_result = self.grayscale_morphology_segmentation(prep_gray)
        elif self.SEG_METHOD == SEGMENTATION_METHOD.KMEANS_RGB:
            seg_result = self.kmeans_segmentation_rgb(img)
            # Convert to binary mask (example: select cluster 0)
            seg_result = (seg_result == 0).astype(np.uint8) * 255
        elif self.SEG_METHOD == SEGMENTATION_METHOD.KMEANS_LBP:
            seg_result = self.kmeans_segmentation_with_lbp(img)
            # Convert to binary mask (example: select cluster 0)
            seg_result = (seg_result == 0).astype(np.uint8) * 255
        else:
            raise ValueError("Unsupported segmentation method.")

        # Postprocessing
        seg_result = self.postprocessing(seg_result)
        
        self.postfix = f"result_" + self.postfix
        self.log += f"Segmentation method: {self.SEG_METHOD.name}\n"
        self.save_result(seg_result)

        return seg_result

    def run_pipeline(self, img_name):
        
        self.postfix = ""
        self.log = ""

        img_key = img_name.split(".")[0]
        save_folder = os.path.join(IMG_FOLDER, img_key)
        self.save_folder = save_folder

        os.makedirs(save_folder, exist_ok=True)

        img_path = os.path.join(IMG_FOLDER, img_name)
        img = Pipeline.load_image(img_path)

        self.orig_img = img
        seg_result = self.run_segmentation_pipeline(img)


def experiment_1():
    img_name = "1.png"

    pipelines = []

    # Preprocessing parameters
    blur_kernel_sizes = [(3, 3), (5, 5)]
    clahe_clip_limits = [2.0, 4.0]
    clahe_tile_sizes = [(8, 8), (16, 16)]

    # -------------------------------------
    # Morphological Segmentation Parameters
    # -------------------------------------
    morph_kernel_sizes = [3, 5]
    morph_iterations_list = [1, 2]
    morph_elements = [cv2.MORPH_ELLIPSE, cv2.MORPH_RECT]

    for blur_ksize in blur_kernel_sizes:
        for clahe_clip in clahe_clip_limits:
            for clahe_tile in clahe_tile_sizes:
                for kernel_size in morph_kernel_sizes:
                    for iteration in morph_iterations_list:
                        for element in morph_elements:
                            p = Pipeline(
                                morph_kernel_size=kernel_size,
                                morph_iterations=iteration,
                                morph_element=element,
                                blur_kernel_size=blur_ksize,
                                clahe_clip_limit=clahe_clip,
                                clahe_tile_size=clahe_tile,
                                seg_method=SEGMENTATION_METHOD.GRAYSCALE_MORPHOLOGY
                            )
                            pipelines.append(p)

    # -------------------------------------
    # KMeans RGB Segmentation Parameters
    # -------------------------------------
    kmeans_clusters_list = [2, 3]
    kmeans_inits = ['k-means++', 'random']

    for blur_ksize in blur_kernel_sizes:
        for clahe_clip in clahe_clip_limits:
            for clahe_tile in clahe_tile_sizes:
                for clusters in kmeans_clusters_list:
                    for init_method in kmeans_inits:
                        p = Pipeline(
                            kmeans_clusters=clusters,
                            kmeans_init=init_method,
                            blur_kernel_size=blur_ksize,
                            clahe_clip_limit=clahe_clip,
                            clahe_tile_size=clahe_tile,
                            seg_method=SEGMENTATION_METHOD.KMEANS_RGB
                        )
                        pipelines.append(p)

    # -------------------------------------
    # KMeans LBP Segmentation Parameters
    # -------------------------------------
    lbp_radius_list = [1, 2]
    lbp_methods = ['uniform', 'nri_uniform']

    for blur_ksize in blur_kernel_sizes:
        for clahe_clip in clahe_clip_limits:
            for clahe_tile in clahe_tile_sizes:
                for radius in lbp_radius_list:
                    points = 8 * radius
                    for lbp_method in lbp_methods:
                        p = Pipeline(
                            kmeans_lbp_radius=radius,
                            kmeans_lbp_points=points,
                            kmeans_lbp_method=lbp_method,
                            blur_kernel_size=blur_ksize,
                            clahe_clip_limit=clahe_clip,
                            clahe_tile_size=clahe_tile,
                            seg_method=SEGMENTATION_METHOD.KMEANS_LBP
                        )
                        pipelines.append(p)

    # Now run all pipelines
    for pipeline_obj in pipelines:
        pipeline_obj:Pipeline
        pipeline_obj.run_pipeline(img_name)


def full_experiment():

    img_paths = [
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png",
        "6.png"
    ]

    pipelines = []

    # Preprocessing parameters
    blur_kernel_sizes = [(3, 3), (5, 5)]
    clahe_clip_limits = [2.0, 4.0]
    clahe_tile_sizes = [(8, 8), (16, 16)]

    # -------------------------------------
    # Morphological Segmentation Parameters
    # -------------------------------------
    morph_kernel_sizes = [3, 5]
    morph_iterations_list = [1, 2]
    morph_elements = [cv2.MORPH_ELLIPSE, cv2.MORPH_RECT]

    for blur_ksize in blur_kernel_sizes:
        for clahe_clip in clahe_clip_limits:
            for clahe_tile in clahe_tile_sizes:
                for kernel_size in morph_kernel_sizes:
                    for iteration in morph_iterations_list:
                        for element in morph_elements:
                            p = Pipeline(
                                morph_kernel_size=kernel_size,
                                morph_iterations=iteration,
                                morph_element=element,
                                blur_kernel_size=blur_ksize,
                                clahe_clip_limit=clahe_clip,
                                clahe_tile_size=clahe_tile,
                                seg_method=SEGMENTATION_METHOD.GRAYSCALE_MORPHOLOGY
                            )
                            pipelines.append(p)

    # -------------------------------------
    # KMeans RGB Segmentation Parameters
    # -------------------------------------
    kmeans_clusters_list = [2, 3]
    kmeans_inits = ['k-means++', 'random']

    for blur_ksize in blur_kernel_sizes:
        for clahe_clip in clahe_clip_limits:
            for clahe_tile in clahe_tile_sizes:
                for clusters in kmeans_clusters_list:
                    for init_method in kmeans_inits:
                        p = Pipeline(
                            kmeans_clusters=clusters,
                            kmeans_init=init_method,
                            blur_kernel_size=blur_ksize,
                            clahe_clip_limit=clahe_clip,
                            clahe_tile_size=clahe_tile,
                            seg_method=SEGMENTATION_METHOD.KMEANS_RGB
                        )
                        pipelines.append(p)

    # -------------------------------------
    # KMeans LBP Segmentation Parameters
    # -------------------------------------
    lbp_radius_list = [1, 2]
    lbp_methods = ['uniform', 'nri_uniform']

    for blur_ksize in blur_kernel_sizes:
        for clahe_clip in clahe_clip_limits:
            for clahe_tile in clahe_tile_sizes:
                for radius in lbp_radius_list:
                    points = 8 * radius
                    for lbp_method in lbp_methods:
                        p = Pipeline(
                            kmeans_lbp_radius=radius,
                            kmeans_lbp_points=points,
                            kmeans_lbp_method=lbp_method,
                            blur_kernel_size=blur_ksize,
                            clahe_clip_limit=clahe_clip,
                            clahe_tile_size=clahe_tile,
                            seg_method=SEGMENTATION_METHOD.KMEANS_LBP
                        )
                        pipelines.append(p)

    # Now run all pipelines
    for img_name in img_paths:
        for pipeline_obj in pipelines:
            pipeline_obj:Pipeline
            pipeline_obj.run_pipeline(img_name)

    

if __name__ == "__main__":
    # Example: Perform morphology-based segmentation
    full_experiment()
