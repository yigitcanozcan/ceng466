# CENG 466 THE2
# Mert Uludoğan 2380996
# Yiğitcan Özcan 2521847

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
from scipy.stats import entropy  # This is used for KL divergence
from PIL import Image, ImageChops

### COMMON FUNCTIONS ###
input_folder = 'THE2_Images/Question1/'
output_folder = 'THE2_Images/Question1/'


def read_image(filename, gray_scale=False):
    # CV2 is just a suggestion you can use other libraries as well
    if gray_scale:
        img = cv2.imread(input_folder + filename, cv2.IMREAD_GRAYSCALE)
        return img
    img = cv2.imread(input_folder + filename)
    return img


def write_image(img, filename):
    # CV2 is just a suggestion you can use other libraries as well
    if os.path.exists(output_folder) == False:
        os.makedirs(output_folder)
    cv2.imwrite(output_folder + filename, img)

#QUESTION 1
# Load the images
img1 = read_image("a1.png", gray_scale=True)
img2 = read_image("a2.png", gray_scale=True)

# Save grayscale images
write_image(img1, "grayscale_a1.png")
write_image(img2, "grayscale_a2.png",)

# Define custom kernels
roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

def apply_filters(img, prefix):
    # Sobel
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)

    # Roberts
    roberts = cv2.filter2D(img, -1, roberts_x) + cv2.filter2D(img, -1, roberts_y)

    # Prewitt
    prewitt = cv2.filter2D(img, -1, prewitt_x) + cv2.filter2D(img, -1, prewitt_y)

    # Save outputs
    write_image(sobel, f"{prefix}_sobel.png")
    write_image(roberts, f"{prefix}_roberts.png")
    write_image(prewitt, f"{prefix}_prewitt.png")

apply_filters(img1, "a1")
apply_filters(img2, "a2")                     #step 1 and step 2 completed.

def blur_images(img, prefix):
    for k in [3, 5, 7]:
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        write_image(blurred, f"{prefix}_blurred_k{k}.png")

blur_images(img1, "a1")
blur_images(img2, "a2")                         #step 3

#step 4
for k in [3, 5, 7]:
    img1_blurred = read_image(f"a1_blurred_k{k}.png", gray_scale=True)
    img2_blurred = read_image(f"a2_blurred_k{k}.png", gray_scale=True)
    apply_filters(img1_blurred, f"a1_blurred_k{k}_filtered")
    apply_filters(img2_blurred, f"a2_blurred_k{k}_filtered")

#step 5
def binarize_msb(img, prefix):
    msb_img = ((img >> 7) & 1) * 255  # Extract MSB and scale to 0-255
    write_image(msb_img, f"{prefix}_msb.jpg")
    return msb_img

msb_img1 = binarize_msb(img1, "a1")
msb_img2 = binarize_msb(img2, "a2")

#step 6
apply_filters(msb_img1, "a1_msb")
apply_filters(msb_img2, "a2_msb")

#step 7
blur_images(msb_img1, "a1_msb")
blur_images(msb_img2, "a2_msb")

#step 8
for k in [3, 5, 7]:
    img1_msb_blurred = read_image(f"a1_msb_blurred_k{k}.png", gray_scale=True)
    img2_msb_blurred = read_image(f"a2_msb_blurred_k{k}.png", gray_scale=True)
    apply_filters(img1_msb_blurred, f"a1_msb_blurred_k{k}")
    apply_filters(img2_msb_blurred, f"a2_msb_blurred_k{k}")
