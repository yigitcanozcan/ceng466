import numpy as np
import cv2
import matplotlib.pyplot as  plt
import math
import sys, time
from PIL import Image, ImageChops

from PIL.ImageOps import scale

input_folder = 'THE1_Images/'
output_folder = 'THE1_Outputs/'


# https://github.com/yunabe/codelab/blob/master/misc/terminal_progressbar/progress.py
def get_progressbar_str(progress):
    END = 170
    MAX_LEN = 30
    BAR_LEN = int(MAX_LEN * progress)
    return ('Progress:[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) +
            '] %.1f%%' % (progress * 100.))


def read_image(filename, gray_scale = False):
    # CV2 is just a suggestion you can use other libraries as well
    if gray_scale:
        img = cv2.imread(input_folder + filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        return img
    img = cv2.imread(input_folder + filename)
    return img

def write_image(img, filename):
    # CV2 is just a suggestion you can use other libraries as well
    cv2.imwrite(output_folder+filename, img)

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

'''
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
'''

# Interpolation kernel
def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

#Padding
def padding(img,H,W,C):
    zimg = np.zeros((H+4,W+4,C))
    zimg[2:H+2,2:W+2,:C] = img
    #Pad the first/last two col and row
    zimg[2:H+2,0:2,:C]=img[:,0:1,:C]
    zimg[H+2:H+4,2:W+2,:]=img[H-1:H,:,:]
    zimg[2:H+2,W+2:W+4,:]=img[:,W-1:W,:]
    zimg[0:2,2:W+2,:C]=img[0:1,:,:C]
    #Pad the missing eight points
    zimg[0:2,0:2,:C]=img[0,0,:C]
    zimg[H+2:H+4,0:2,:C]=img[H-1,0,:C]
    zimg[H+2:H+4,W+2:W+4,:C]=img[H-1,W-1,:C]
    zimg[0:2,W+2:W+4,:C]=img[0,W-1,:C]
    return zimg

def bicubic_rotate(img, degree, a=-0.5):
    # Get image size
    H, W, C = img.shape

    # Padding the image to handle edges
    img = padding(img, H, W, C)
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
                x_new = v * cos_angle + w * sin_angle + cx
                y_new = -v * sin_angle + w * cos_angle + cy

                # Bicubic interpolation
                if 1 <= x_new <= W and 1 <= y_new <= H:
                    x1 = 1 + x_new - math.floor(x_new)
                    x2 = x_new - math.floor(x_new)
                    x3 = math.floor(x_new) + 1 - x_new
                    x4 = math.floor(x_new) + 2 - x_new

                    y1 = 1 + y_new - math.floor(y_new)
                    y2 = y_new - math.floor(y_new)
                    y3 = math.floor(y_new) + 1 - y_new
                    y4 = math.floor(y_new) + 2 - y_new

                    mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
                    mat_m = np.matrix([[img[round(y_new - y1), round(x_new - x1), c], img[round(y_new - y2), round(x_new - x1), c], img[round(y_new + y3), round(x_new - x1), c], img[round(y_new + y4), round(x_new - x1), c]],
                                       [img[round(y_new - y1), round(x_new - x2), c], img[round(y_new - y2), round(x_new - x2), c], img[round(y_new + y3), round(x_new - x2), c], img[round(y_new + y4), round(x_new - x2), c]],
                                       [img[round(y_new - y1), round(x_new + x3), c], img[round(y_new - y2), round(x_new + x3), c], img[round(y_new + y3), round(x_new + x3), c], img[round(y_new + y4), round(x_new + x3), c]],
                                       [img[round(y_new - y1), round(x_new + x4), c], img[round(y_new - y2), round(x_new + x4), c], img[round(y_new + y3), round(x_new + x4), c], img[round(y_new + y4), round(x_new + x4), c]]])

                    mat_r = np.matrix([[u(y1,a)], [u(y2,a)], [u(y3,a)], [u(y4,a)]])
                    rotated_img[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)


                    # Print progress
                    inc = inc + 1
                    sys.stderr.write('\r\033[K' + get_progressbar_str(inc / (C * H * W)))
                    sys.stderr.flush()
    sys.stderr.write('\n')
    sys.stderr.flush()
    print("Rotation completed.")
    return rotated_img

'''
def bicubic_scale(img, scale, a=-0.5):
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

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
                mat_m = np.matrix([[img[round(y - y1), round(x - x1), c], img[round(y - y2), round(x - x1), c], img[round(y + y3), round(x - x1), c], img[round(y + y4), round(x - x1), c]],
                                   [img[round(y - y1), round(x - x2), c], img[round(y - y2), round(x - x2), c], img[round(y + y3), round(x - x2), c], img[round(y + y4), round(x - x2), c]],
                                   [img[round(y - y1), round(x + x3), c], img[round(y - y2), round(x + x3), c], img[round(y + y3), round(x + x3), c], img[round(y + y4), round(x + x3), c]],
                                   [img[round(y - y1), round(x + x4), c], img[round(y - y2), round(x + x4), c], img[round(y + y3), round(x + x4), c], img[round(y + y4), round(x + x4), c]]])

                mat_r = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
                scaled_img[j, i, c] = np.dot(np.dot(mat_l, mat_m),mat_r)

                inc = inc + 1
                sys.stderr.write('\r\033[K' + get_progressbar_str(inc / (C * dH * dW)))
                sys.stderr.flush()
    sys.stderr.write('\n')
    sys.stderr.flush()

    print("Scaling completed.")
    return scaled_img
'''

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
    # Find centers of both images
    #center_y1, center_x1 = img1.shape[0] // 2, img1.shape[1] // 2
    #center_y2, center_x2 = img2.shape[0] // 2, img2.shape[1] // 2

    # Determine the smaller dimensions
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    # Calculate coordinates to crop around the top left
    y_start = 0
    y_end = min_height-1
    x_start = 0
    x_end = min_width-1

    # Crop images to the common top left region
    cropped_img1 = img1[y_start:y_end, x_start:x_end]
    cropped_img2 = img2[y_start:y_end, x_start:x_end]

    # Compute mean squared difference
    distance = np.mean((cropped_img1 - cropped_img2) ** 2)
    return distance

if __name__ == '__main__':
    ###################### Q1
    # Read original image
    img_original = read_image('q1_1.png')
    # Read corrupted image
    img = read_image('ratio_4_degree_30.png')
    # Correct the image with linear interpolation
    corrected_img_linear = rotate_upsample(img, 4, 30, 'linear')
    write_image(corrected_img_linear, 'q1_1_corrected_linear.png')

    # Correct the image with cubic interpolation
    corrected_img_cubic = rotate_upsample(img, 4, 30, 'cubic')
    write_image(corrected_img_cubic, 'q1_1_corrected_cubic.png')

    # Report the distances
    print("Distance 1:")
    print('The distance between original image and image corrected with linear interpolation is ', compute_distance(img_original, corrected_img_linear))
    print('The distance between original image and image corrected with cubic interpolation is ', compute_distance(img_original, corrected_img_cubic))

    # Repeat the same steps for the second image
    img_original = read_image('q1_2.png')
    img = read_image('ratio_8_degree_45.png')
    corrected_img_linear = rotate_upsample(img, 8, 45, 'linear')
    write_image(corrected_img_linear, 'q1_2_corrected_linear.png')
    corrected_img_cubic = rotate_upsample(img, 8, 45, 'cubic')
    write_image(corrected_img_cubic, 'q1_2_corrected_cubic.png')

    # Report the distances
    print("Distance 2:")
    print('The distance between original image and image corrected with linear interpolation is ', compute_distance(img_original, corrected_img_linear))
    print('The distance between original image and image corrected with cubic interpolation is ', compute_distance(img_original, corrected_img_cubic))
