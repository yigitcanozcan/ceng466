# CENG 466 THE2
# Mert Uludoğan 2380996
# Yiğitcan Özcan 2521847

import numpy as np
import cv2
import os

### COMMON FUNCTIONS ###
input_folder = 'THE2_Images/Question2/'
output_folder = 'THE2_Images/Question2/'


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

#QUESTION 2
# Load the images
img1 = read_image("b1.jpg")
img2 = read_image("b2.jpg")
img3 = read_image("b3.jpg")


blue_channel, green_channel, red_channel = cv2.split(img1)
write_image(blue_channel, "b1_blue_channel.png")
write_image(green_channel, "b1_green_channel.png")
write_image(red_channel, "b1_red_channel.png")

b1_gaus_blurred = cv2.GaussianBlur(img1, (11, 11), 0)
b1_med_blurred = cv2.medianBlur(img1, 7)
write_image(b1_med_blurred, "b1_median.png")
write_image(b1_gaus_blurred, "b1_gaussian.png")


blue_channel, green_channel, red_channel = cv2.split(img2)
write_image(blue_channel, "b2_blue_channel.png")
write_image(green_channel, "b2_green_channel.png")
write_image(red_channel, "b2_red_channel.png")

b2_gaus_blurred = cv2.GaussianBlur(img2, (11, 11), 0)
b2_med_blurred = cv2.medianBlur(img2, 7)
write_image(b2_med_blurred, "b2_median.png")
write_image(b2_gaus_blurred, "b2_gaussian.png")


blue_channel, green_channel, red_channel = cv2.split(img3)
write_image(blue_channel, "b3_blue_channel.png")
write_image(green_channel, "b3_green_channel.png")
write_image(red_channel, "b3_red_channel.png")

b3_gaus_blurred = cv2.GaussianBlur(img3, (11, 11), 0)
b3_med_blurred = cv2.medianBlur(img3, 7)
write_image(b3_med_blurred, "b3_median.png")
write_image(b3_gaus_blurred, "b3_gaussian.png")

# It might be necessary to apply different filters to different channels, since channels do not have the same type of noise.

def apply_fourier_filters(img, output_prefix):
    # Split the image into channels
    b_channel, g_channel, r_channel = cv2.split(img)

    # Function to apply Fourier filters on a single channel
    def filter_channel(channel, prefix):
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # Fourier Transform
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)

        # Frequency grid
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        u = u - ccol
        v = v - crow
        distance = np.sqrt(u**2 + v**2)

        # Ideal Low-Pass Filter (ILP)
        cutoff_low = 50
        ilp_filter = (distance <= cutoff_low).astype(np.float32)

        # Band-Pass Filter (BP)
        cutoff_low_bp, cutoff_high_bp = 30, 70
        bp_filter = ((distance >= cutoff_low_bp) & (distance <= cutoff_high_bp)).astype(np.float32)

        # Band-Reject Filter (BR)
        cutoff_low_br, cutoff_high_br = 30, 70
        br_filter = ((distance < cutoff_low_br) | (distance > cutoff_high_br)).astype(np.float32)

        # Apply filters
        ilp_result = dft_shift * ilp_filter
        bp_result = dft_shift * bp_filter
        br_result = dft_shift * br_filter

        # Transform back to spatial domain
        ilp_img = np.abs(np.fft.ifft2(np.fft.ifftshift(ilp_result)))
        bp_img = np.abs(np.fft.ifft2(np.fft.ifftshift(bp_result)))
        br_img = np.abs(np.fft.ifft2(np.fft.ifftshift(br_result)))

        # Normalize results
        ilp_img = cv2.normalize(ilp_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        bp_img = cv2.normalize(bp_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        br_img = cv2.normalize(br_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return ilp_img, bp_img, br_img

    # Apply filters to each channel
    b_ilp, b_bp, b_br = filter_channel(b_channel, "b")
    g_ilp, g_bp, g_br = filter_channel(g_channel, "g")
    r_ilp, r_bp, r_br = filter_channel(r_channel, "r")

    # Merge results for each filter
    ilp_result = cv2.merge((b_ilp, g_ilp, r_ilp))
    bp_result = cv2.merge((b_bp, g_bp, r_bp))
    br_result = cv2.merge((b_br, g_br, r_br))

    # Save the results
    write_image(ilp_result, f"{output_prefix}_ilp.png")
    write_image(bp_result, f"{output_prefix}_bp.png")
    write_image(br_result, f"{output_prefix}_br.png")


apply_fourier_filters(img1, "b1")
apply_fourier_filters(img2, "b2")
apply_fourier_filters(img3, "b3")
