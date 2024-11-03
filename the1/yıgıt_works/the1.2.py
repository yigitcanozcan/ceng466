import numpy as np
import cv2
import matplotlib.pyplot as  plt
import math

input_folder = 'THE1_Images/'
output_folder = 'THE1_Outputs/'


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

def hue_calc(img):
    # divide B G R channels
    blue = img[:, :, 0].astype(float) / 255  # Convert to float and normalize
    green = img[:, :, 1].astype(float) / 255
    red = img[:, :, 2].astype(float) / 255

    hue_img = np.copy(blue)  # Just initialize it with the shape of blue
    height = img.shape[0]
    width = img.shape[1]

    for j in range(height):
        for i in range(width):
            numerator = 0.5 * ((red[j][i] - green[j][i]) + (red[j][i] - blue[j][i]))
            denominator = math.sqrt((red[j][i] - green[j][i]) ** 2 + (red[j][i] - blue[j][i]) * (green[j][i] - blue[j][i]))
            # Directly applying the formula caused math domain error because of the acos function
            if denominator == 0:
                hue_img[j][i] = 0  # Handle divide by zero case, e.g., gray colors
            else:
                hue_value = numerator / denominator

                # Clamp hue_value to the range [-1, 1] to avoid domain error in acos
                hue_value = max(min(hue_value, 1), -1)

                hue_img[j][i] = math.acos(hue_value)
    return hue_img

def histogram_calc(hue, num_bins=430):
    # Convert hue from radians to degrees for better readability if needed
    hue_degrees = hue * (180 / np.pi)

    # Calculate the histogram
    hist, bin_edges = np.histogram(hue_degrees, bins=num_bins, range=(0, 180))

    # Optional: Normalize the histogram
    hist = hist / np.sum(hist)
    return hist, bin_edges

def plot_histogram(hist, bins):
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), edgecolor="black")
    plt.xlabel("Hue (degrees)")
    plt.ylabel("Frequency")
    plt.title("Hue Histogram")
    plt.show()

def kl_divergence(hue, image):
    # Step 1: Calculate the hue matrix of the image
    image_hue = hue_calc(image)

    # Step 2: Calculate histograms for both the hue matrix and image hue
    hue_hist, _ = histogram_calc(hue)
    image_hue_hist, _ = histogram_calc(image_hue)

    P = hue_hist
    Q = image_hue_hist

    # Step 4: Calculate KL divergence (add small value to avoid log(0))
    epsilon = 1e-10  # A small constant to avoid division by zero
    P = P + epsilon
    Q = Q + epsilon

    kl_div = np.sum(P * np.log(P / Q))

    return kl_div

def desert_or_forest(img):
    '''img: image to be classified as desert or forest
    return a string: either 'desert'  or 'forest'

    You should compare the KL Divergence between histograms of hue channel. Please provide images and discuss these histograms in your report'''
    desert1 = read_image('desert1.jpg')
    desert2 = read_image('desert2.jpg')
    forest1 = read_image('forest1.jpg')
    forest2 = read_image('forest2.jpg')

    hue_img = hue_calc(img)
    dd1 = kl_divergence(hue_img, desert1)
    dd2 = kl_divergence(hue_img, desert2)
    fd1 = kl_divergence(hue_img, forest1)
    fd2 = kl_divergence(hue_img, forest2)

    if dd1 < fd1 and dd1 < fd2 and dd2 < fd1 and dd2 < fd2:
        return "desert"
    elif fd1 < dd1 and fd1 < dd2 and fd2 < dd1 and fd2 < dd2:
        return "forest"
    else:
        return "Cannot decide whether it is desert or forest."

if __name__ == '__main__':
    ###################### Q2
    img = read_image('q2_1.jpg')
    result = desert_or_forest(img)
    print("Given image q2_1 is an image of a ", result)

    img = read_image('q2_2.jpg')
    result = desert_or_forest(img)
    print("Given image q2_2 is an image of a ", result)