# CENG 466 THE1
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
input_folder = 'THE1_Images/'
output_folder = 'THE1_Outputs/'


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


#### Question 1 ####
def bilinear_rotate(img, degree):
    # Get image dimensions
    height, width = img.shape[:2]

    # Calculate center of the image
    cx, cy = width // 2, height // 2

    # Create an empty image for the rotated result
    rotated_img = np.zeros_like(img)

    # Convert the angle to radians
    alpha = np.radians(degree)

    # Inverse rotation matrix (backward transformation)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    for y in range(height):
        for x in range(width):
            # Translate coordinates to center
            v, w = x - cx, y - cy

            # Apply backward rotation
            x_prime = v * cos_alpha + w * sin_alpha + cx
            y_prime = -v * sin_alpha + w * cos_alpha + cy

            # Check if the transformed coordinates are within bounds
            if 0 <= x_prime < width and 0 <= y_prime < height:
                # Get integer parts (top-left corner)
                x1 = int(np.floor(x_prime))
                y1 = int(np.floor(y_prime))

                # Ensure the integer coordinates are valid
                if x1 < 0 or x1 >= width - 1 or y1 < 0 or y1 >= height - 1:
                    continue

                # Get the fractional parts
                a = x_prime - x1
                b = y_prime - y1

                # Perform bilinear interpolation
                rotated_img[y, x] = (1 - a) * (1 - b) * img[y1, x1] + \
                                    a * (1 - b) * img[y1, x1 + 1] + \
                                    (1 - a) * b * img[y1 + 1, x1] + \
                                    a * b * img[y1 + 1, x1 + 1]

    return rotated_img

def bilinear_scale(img, scale):
    # Normal scaling takes too much time, so I used vectorized version of my code. WHY????

    img_height, img_width, channels = img.shape
    height = int(img_height * scale)
    width = int(img_width * scale)

    # Create a new empty array for the resized image
    scaled_img = np.zeros((height, width, channels), dtype=img.dtype)

    # Compute the ratio between original and target image sizes
    x_ratio = (img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = (img_height - 1) / (height - 1) if height > 1 else 0

    # Generate grids of coordinates for interpolation
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate the corresponding positions in the original image
    x_orig = x_ratio * x_grid
    y_orig = y_ratio * y_grid

    # Get the floor and ceiling coordinates
    x_l = np.floor(x_orig).astype(int)
    x_h = np.ceil(x_orig).astype(int)
    y_l = np.floor(y_orig).astype(int)
    y_h = np.ceil(y_orig).astype(int)

    # Calculate the weights for interpolation
    x_weight = x_orig - x_l
    y_weight = y_orig - y_l

    # Clip to prevent out-of-bounds access
    x_l = np.clip(x_l, 0, img_width - 1)
    x_h = np.clip(x_h, 0, img_width - 1)
    y_l = np.clip(y_l, 0, img_height - 1)
    y_h = np.clip(y_h, 0, img_height - 1)

    # Perform bilinear interpolation
    for ch in range(channels):
        # Get pixel values at the 4 surrounding points
        a = img[y_l, x_l, ch]
        b = img[y_l, x_h, ch]
        c = img[y_h, x_l, ch]
        d = img[y_h, x_h, ch]

        # Interpolate
        scaled_img[:, :, ch] = (a * (1 - x_weight) * (1 - y_weight) +
                             b * x_weight * (1 - y_weight) +
                             c * y_weight * (1 - x_weight) +
                             d * x_weight * y_weight)

    return scaled_img


# Cubic convolution function
def u(s,a):
    s = abs(s)
    if s<=1:
        return (a+2)*(s**3)-(a+3)*(s**2)+1
    elif (s>1) and (s<=2):
        return a*(s**3)-(5*a)*(s**2)+(8*a)*s-4*a
    return 0

#Padding
def padding(img,H,W,C):
    pad = np.zeros((H + 4, W + 4, C))
    pad[2:H + 2, 2:W + 2, :C] = img  # Place the original image in the center
    # Pad the edges
    for i in range(2):
        # Left and Right sides
        pad[2:H + 2, i, :C] = img[:, 0, :C]  # Left columns
        pad[2:H + 2, W + 2 + i, :C] = img[:, W - 1, :C]  # Right columns
        # Top and Bottom sides
        pad[i, 2:W + 2, :C] = img[0, :, :C]  # Top rows
        pad[H + 2 + i, 2:W + 2, :C] = img[H - 1, :, :C]  # Bottom rows

    # Pad the corners
    corners = [ (0, 0), (0, W + 2),  # Top-left and Top-right
                (H + 2, 0), (H + 2, W + 2) ]  # Bottom-left and Bottom-right
    for (y, x) in corners:
        pad[y:y + 2, x:x + 2, :C] = img[0 if y == 0 else H - 1, 0 if x == 0 else W - 1, :C]

    return pad

def bicubic_rotate(img, degree, a=-0.75):
    # Get image size
    H, W, C = img.shape

    # Padding the image to handle edges
    img = padding(img,H, W, C)
    # Calculate the center of the original image
    cx, cy = W // 2, H // 2

    # Create new image for rotated result
    rotated_img = np.zeros((H, W, C))

    # Rotation parameters
    rad = np.radians(degree)
    cos_angle = np.cos(rad)
    sin_angle = np.sin(rad)

    inc = 0
    for c in range(C):
        for j in range(H):
            for i in range(W):
                v, w = i - cx, j - cy
                # Backward rotation to get the original coordinates
                x = v * cos_angle + w * sin_angle + cx
                y = -v * sin_angle + w * cos_angle + cy

                # Bicubic interpolation
                if 1 <= x <= W and 1 <= y <= H:
                    x1 = 1 + x - math.floor(x)
                    x2 = x - math.floor(x)
                    x3 = math.floor(x) + 1 - x
                    x4 = math.floor(x) + 2 - x
                    x1_new = round(x - x1)
                    x2_new = round(x - x2)
                    x3_new = round(x + x3)
                    x4_new = round(x + x4)

                    y1 = 1 + y - math.floor(y)
                    y2 = y - math.floor(y)
                    y3 = math.floor(y) + 1 - y
                    y4 = math.floor(y) + 2 - y
                    y1_new = round(y - y1)
                    y2_new = round(y - y2)
                    y3_new = round(y + y3)
                    y4_new = round(y + y4)

                    mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
                    mat_m = np.matrix([[img[y1_new, x1_new, c], img[y2_new, x1_new, c], img[y3_new, x1_new, c], img[y4_new, x1_new, c]],
                                       [img[y1_new, x2_new, c], img[y2_new, x2_new, c], img[y3_new, x2_new, c], img[y4_new, x2_new, c]],
                                       [img[y1_new, x3_new, c], img[y2_new, x3_new, c], img[y3_new, x3_new, c], img[y4_new, x3_new, c]],
                                       [img[y1_new, x4_new, c], img[y2_new, x4_new, c], img[y3_new, x4_new, c], img[y4_new, x4_new, c]]])

                    mat_r = np.matrix([[u(y1,a)], [u(y2,a)], [u(y3,a)], [u(y4,a)]])
                    rotated_img[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

    return rotated_img

def bicubic_scale(img, scale, a=-0.75):
    #Get image size
    H,W,C = img.shape

    img = padding(img,H,W,C)
    #Create new image
    dH = math.floor(H*scale)
    dW = math.floor(W*scale)
    scaled_img = np.zeros((dH, dW, 3))

    h = 1/scale

    inc = 0
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                x, y = i * h + 2 , j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x
                x1_new = round(x - x1)
                x2_new = round(x - x2)
                x3_new = round(x + x3)
                x4_new = round(x + x4)

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y
                y1_new = round(y - y1)
                y2_new = round(y - y2)
                y3_new = round(y + y3)
                y4_new = round(y + y4)

                mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
                mat_m = np.matrix([[img[y1_new, x1_new, c], img[y2_new, x1_new, c], img[y3_new, x1_new, c], img[y4_new, x1_new, c]],
                                   [img[y1_new, x2_new, c], img[y2_new, x2_new, c], img[y3_new, x2_new, c], img[y4_new, x2_new, c]],
                                   [img[y1_new, x3_new, c], img[y2_new, x3_new, c], img[y3_new, x3_new, c], img[y4_new, x3_new, c]],
                                   [img[y1_new, x4_new, c], img[y2_new, x4_new, c], img[y3_new, x4_new, c], img[y4_new, x4_new, c]]])

                mat_r = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
                scaled_img[j, i, c] = np.dot(np.dot(mat_l, mat_m),mat_r)

    return scaled_img

# For trimming the edges after backward rotation. In order to prevent aggressive trimming we add a threshold value also.
def trim(im, bg_color=(0, 0, 0), threshold=20):
    # Ensure the image is in uint8 format
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    # Convert the NumPy array (BGR format) to a PIL Image (RGB format)
    im_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # Create a background image filled with the specified background color
    bg = Image.new(im_pil.mode, im_pil.size, bg_color)

    # Calculate the difference and then adjust it with a threshold
    diff = ImageChops.difference(im_pil, bg)
    diff = ImageChops.add(diff, diff, 2.0, -threshold)  # Adjust threshold here
    bbox = diff.getbbox()

    # If there is a bounding box, crop the image
    if bbox:
        cropped_image = im_pil.crop(bbox)

        # Convert back to NumPy array (BGR format)
        return cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
    else:
        # Return the original image if no cropping is needed
        return im



def rotate_upsample(img, scale, degree, interpolation_type):
    if interpolation_type == 'linear':
        rotated_img = bilinear_rotate(img, degree)
        scaled_img = cv2.resize(rotated_img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_LINEAR)
        scaled_img = trim(scaled_img)

    elif interpolation_type == 'cubic':
        rotated_img = bicubic_rotate(img, degree)
        scaled_img = cv2.resize(rotated_img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        scaled_img = trim(scaled_img)

    else:
        raise ValueError("Invalid interpolation type")

    return scaled_img




def compute_distance(img1, img2):
    if img1.shape == img2.shape:
        return np.mean((img1 - img2) ** 2)

    # Determine the smaller dimensions
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    # Calculate coordinates to crop around the top left
    y_start = 0
    y_end = min_height - 1
    x_start = 0
    x_end = min_width - 1

    # Crop images to the common top left region
    cropped_img1 = img1[y_start:y_end, x_start:x_end]
    cropped_img2 = img2[y_start:y_end, x_start:x_end]

    # Compute mean squared difference
    distance = np.mean((cropped_img1 - cropped_img2) ** 2)
    return distance


#### Question 2 ####
def calculate_histogram(channel, bins=50, range_max=180):
    # Flatten the channel to a 1D array
    channel_flat = channel.flatten()

    # Initialize the histogram array
    histogram = np.zeros(bins, dtype=float)

    # Calculate bin width
    bin_width = range_max / bins

    # Populate the histogram
    for intensity in channel_flat:
        bin_index = int(intensity // bin_width)  # Get the bin index for this pixel intensity
        histogram[bin_index] += 1

    # Normalize the histogram
    histogram /= channel_flat.size  # Normalize so that sum(histogram) = 1

    return histogram


def compute_histogram(image):
    """Compute the normalized histogram of the hue channel."""
    # Convert image to HSI and extract the hue channel
    hsi_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsi_img[:, :, 0]
    # Compute the histogram of the hue channel
    hist = calculate_histogram(hue_channel)
    return hist


def compute_kl_divergence(hist1, hist2):
    """Compute the KL Divergence between two histograms."""
    # Add a small epsilon to avoid division by zero or log(0)
    epsilon = 1e-10
    hist1 = hist1 + epsilon
    hist2 = hist2 + epsilon
    return entropy(hist1, hist2)


def desert_or_forest(img):
    """Classify the input image as 'desert' or 'forest'."""
    # Load the database images
    desert1 = read_image('desert1.jpg')
    desert2 = read_image('desert2.jpg')
    forest1 = read_image('forest1.jpg')
    forest2 = read_image('forest2.jpg')

    # Compute histograms for the desert and forest images (Hue channel)
    desert_hist1 = compute_histogram(desert1)
    desert_hist2 = compute_histogram(desert2)
    forest_hist1 = compute_histogram(forest1)
    forest_hist2 = compute_histogram(forest2)

    # Compute the histogram for the input image
    input_hist = compute_histogram(img)

    # Compute KL Divergences with desert and forest images
    kl_desert1 = compute_kl_divergence(input_hist, desert_hist1)
    kl_desert2 = compute_kl_divergence(input_hist, desert_hist2)
    kl_forest1 = compute_kl_divergence(input_hist, forest_hist1)
    kl_forest2 = compute_kl_divergence(input_hist, forest_hist2)

    print("kl_desert1: ", kl_desert1)
    print("kl_desert2: ", kl_desert2)
    print("kl_forest1: ", kl_forest1)
    print("kl_forest2: ", kl_forest2)

    # Average the KL divergences for desert and forest images
    avg_kl_desert = (kl_desert1 + kl_desert2) / 2.0
    avg_kl_forest = (kl_forest1 + kl_forest2) / 2.0

    print("avg_kl_desert: ", avg_kl_desert)
    print("avg_kl_forest: ", avg_kl_forest)

    # Classify based on which average KL divergence is smaller
    if avg_kl_desert < avg_kl_forest:
        return 'desert'
    else:
        return 'forest'


def our_example():
    bird1 = read_image('bird1.jpg')
    bird2 = read_image('bird2.jpg')
    bird3 = read_image('bird3.jpg')

    # Compute histograms for the desert and forest images (Hue channel)
    bird1_hist = compute_histogram(bird1)
    bird2_hist = compute_histogram(bird2)
    bird3_hist = compute_histogram(bird3)

    kl_12 = compute_kl_divergence(bird1_hist, bird2_hist)
    kl_13 = compute_kl_divergence(bird1_hist, bird3_hist)

    print("kl_divergence_12: ", kl_12)
    print("kl_divergence_13: ", kl_13)
    if kl_12 < kl_13:
        print('The bird is bird2')
    else:
        print('The bird is bird3')


#### Question 3 ####
def difference_images(img1, img2, threshold=50, size_percentage=0.0025):
    """
    Masks out the differences between img1 and img2 and applies the mask to img2.
    Assumes that an object is absent in img1 and present in img2.
    Removes small noise based on object size.

    :param img1: The first image (before the object appears)
    :param img2: The second image (after the object appears)
    :param threshold: The threshold value for filtering small differences (default: 50)
    :param min_object_size: Minimum area for an object to be considered valid (default: 500)
    :return: The image2 with the mask applied
    """

    # Compute the absolute difference between the two grayscale images
    diff = cv2.absdiff(img2, img1)

    # Calculate minimum object size based on image area
    image_area = img1.shape[0] * img1.shape[1]  # Total area in pixels
    min_object_size = int(image_area * size_percentage)  # Calculate min size as a percentage of the image area

    # Apply a threshold to filter out small differences (e.g., lighting changes, noise)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Ensure mask is a single-channel image
    if len(mask.shape) == 3:  # Check if mask is 3 channels
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Find contours of the objects in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to store the filtered contours (objects)
    filtered_mask = np.zeros_like(mask)

    # Filter out small contours based on the minimum object size
    for contour in contours:
        if cv2.contourArea(contour) > min_object_size:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Apply morphological closing to fill small gaps in the objects
    kernel = np.ones((3, 3), np.uint8)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)

    # Apply the cleaned mask to img2 (highlighting the new object)
    result = cv2.bitwise_and(img2, img2, mask=filtered_mask)

    return result


#### Execution ####
def main():
    def Q1():
        ###################### Q1
        # Read original image
        # Read corrupted image
        def pipline(
                original_image_path,
                corrupted_image_path,
                scale,
                rotation
        ):
            original_name = original_image_path.split('.')[0]
            corrupted_name = corrupted_image_path.split('.')[0]
            img_original = read_image(original_image_path)
            img_path = corrupted_image_path
            img = read_image(img_path)
            # Correct the image with linear interpolation
            corrected_img_linear = rotate_upsample(
                img=img,
                scale=scale,
                degree=rotation,
                interpolation_type='linear'
            )
            write_image(corrected_img_linear, f'{original_name}_corrected_linear.png')
            # Correct the image with cubic interpolation
            corrected_img_cubic = rotate_upsample(
                img=img,
                scale=scale,
                degree=rotation,
                interpolation_type='cubic'
            )
            write_image(corrected_img_cubic, f'{original_name}_corrected_cubic.png')

            # Report the distances
            print('The distance between original image and image corrected with linear interpolation scratch is ',
                  compute_distance(img_original, corrected_img_linear))
            print('The distance between original image and image corrected with cubic interpolation is ',
                  compute_distance(img_original, corrected_img_cubic))

        print("q1_1 with ratio 4 and degree 30")
        pipline(
            original_image_path='q1_1.png',
            corrupted_image_path='ratio_4_degree_30.png',
            scale=4,
            rotation=30
        )
        print("q1_2 with ratio 8 and degree 45")
        pipline(
            original_image_path='q1_2.png',
            corrupted_image_path='ratio_8_degree_45.png',
            scale=8,
            rotation=45
        )


    def Q2():
        # ###################### Q2
        img = read_image('q2_1.jpg')
        result = desert_or_forest(img)
        print("Given image q2_1 is an image of a", result)

        img = read_image('q2_2.jpg')
        result = desert_or_forest(img)
        print("Given image q2_2 is an image of a", result)

        #our_example()

    def Q3():
        # ###################### Q3
        img1 = read_image('q3_a1.png', gray_scale=True)
        img2 = read_image('q3_a2.png', gray_scale=True)
        result = difference_images(img1, img2)
        write_image(result, 'masked_image_a.png')

        img1 = read_image('q3_b1.png')
        img2 = read_image('q3_b2.png')
        result = difference_images(img1, img2)
        write_image(result, 'masked_image_b.png')

    print("Q1")
    Q1()
    print("--- END OF Q1 ---")
    print("Q2")
    Q2()
    print("--- END OF Q2 ---")
    print("Q3")
    Q3()
    print("--- END OF Q3 ---")


if __name__ == '__main__':
    main()