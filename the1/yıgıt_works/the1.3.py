import numpy as np
import cv2
import matplotlib.pyplot as  plt
import math

input_folder = 'THE1_Images/'
output_folder = 'THE1_Outputs/'


def read_image(filename, gray_scale = False):
    # CV2 is just a suggestion you can use other libraries as well
    if gray_scale:
        img = cv2.imread(input_folder + filename, cv2.IMREAD_GRAYSCALE) # !!!!!!!!!! CHANGE IT LATER !!!!!!!!
        return img
    img = cv2.imread(input_folder + filename)
    return img

def write_image(img, filename):
    # CV2 is just a suggestion you can use other libraries as well
    cv2.imwrite(output_folder+filename, img)

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

if __name__ == '__main__':
    ###################### Q3
    img1 = read_image('q3_a1.png', gray_scale=True)
    img2 = read_image('q3_a2.png', gray_scale=True)
    result = difference_images(img1, img2)
    write_image(result, 'masked_image_a.png')

    img1 = read_image('q3_b1.png')
    img2 = read_image('q3_b2.png')
    result = difference_images(img1, img2)
    write_image(result, 'masked_image_b.png')