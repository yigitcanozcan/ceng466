import numpy as np
import cv2 
import matplotlib.pyplot as  plt
import enum
import os
from scipy.stats import entropy  # This is used for KL divergence

input_folder = 'THE1_Images/'
output_folder = 'THE1_Outputs/'

class Rotation:
    def __init__(self):
        pass
    def rotate_image(self, img, degree):
        '''img: img to be rotated
        degree: degree of rotation in counter clockwise (30 means 30 degree rotation in counter clockwise, -30 means 30 degree rotation in clockwise)
        return rotated image'''
        # Rotate the image
        rad = np.deg2rad(-degree)
        rotation_matrix = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ])
        h, w = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        corners = np.array([
            [-h//2, -w//2],
            [h//2, -w//2],
            [h//2, w//2],
            [-h//2, w//2]
        ])
        new_corners = np.dot(rotation_matrix, corners.T)
        new_corners = new_corners.T

        min_corner = np.min(new_corners, axis=0)
        max_corner = np.max(new_corners, axis=0)
        # Calculate the size of the new image
        new_h = int(np.ceil(max_corner[0] - min_corner[0]))
        new_w = int(np.ceil(max_corner[1] - min_corner[1]))
        
        rotated_img = np.zeros((new_h, new_w, channels), dtype=img.dtype)
        center_y, center_x = new_h//2, new_w//2

        inverse_rotation_matrix = np.linalg.inv(rotation_matrix)

        for i in range(new_h):
            for j in range(new_w):
                # Get the coordinate in the original image space
                x, y = np.dot(inverse_rotation_matrix, [j-center_x, i-center_y])
                x = int(x + w//2)
                y = int(y + h//2)
                
                # Check if the calculated coordinates are within the bounds of the original image
                if 0 <= x < w and 0 <= y < h:
                    rotated_img[i, j] = img[y, x]
        return rotated_img

class Interpolation:
    class INTERPOLATION_TYPE(enum.Enum):
        LINEAR = "linear"
        CUBIC = "cubic"

    class NEIGHBOUR_TYPE(enum.Enum):
        EXACT_POINT = "EXACT_POINT"
        X_DIM_POINT = "X_DIM_POINT"
        Y_DIM_POINT = "Y_DIM_POINT"
        TWO_DIM_POINT = "TWO_DIM_POINT"

    def __init__(self):
        pass

    # General Functions
    def get_reverse_scaled_point(self, point, scale, h, w):
        center_point = tuple(map(lambda x: x + 0.5, point))
        reverse_scaled_center_point = tuple(map(lambda x: x / scale, center_point))
        reverse_scaled_point = tuple(map(lambda x: x - 0.5, reverse_scaled_center_point))

        # Correcting out of bounds points
        reverse_scaled_point = (
            min(max(reverse_scaled_point[0], 0), h - 1),
            min(max(reverse_scaled_point[1], 0), w - 1)
        )
        return reverse_scaled_point

    def calculate_weighted_value(self, img, points, weights):
        final_value = 0

        values = []
        for point in points:
            values.append(img[point[0], point[1]])
        for value, weight in zip(values, weights):
            final_value += value * weight
        return final_value

    def interpolate(self, img, scale):
        raise NotImplementedError

class BilinearInterpolation(Interpolation):
    def __init__(self):
        super().__init__()

    # Bilinear Interpolation Specific Functions
    def get_linear_neighbours(self, point, h, w):
        points = []
        neighbour_type = None

        point_y_int = int(point[0])
        point_x_int = int(point[1])

        for y in range(point_y_int, point_y_int+2):
            for x in range(point_x_int, point_x_int+2):
                y_clamped = min(max(y, 0), h - 1)
                x_clamped = min(max(x, 0), w - 1)
                points.append((y_clamped, x_clamped))
        return points
    
    def calculate_bilinear_weights(self, point, neighbour_points):
        weights = []
        for neighbour in neighbour_points:
            weight_x = self.bilinear_kernel(point[1] - neighbour[1])
            weight_y = self.bilinear_kernel(point[0] - neighbour[0])
            weights.append(weight_x * weight_y)
        return weights

    def bilinear_kernel(self, t):
        t = np.abs(t)
        return 1 - t

    def interpolate(self, img, scale):
        h, w = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        scaled_h = int(h * scale)
        scaled_w = int(w * scale)

        scaled_img = np.zeros((scaled_h, scaled_w, channels), dtype=img.dtype)

        for j in range(scaled_h):
            for i in range(scaled_w):
                point = (j, i)
                q = self.get_reverse_scaled_point(point, scale, h, w)
                neighbour_points = self.get_linear_neighbours(q, h, w)
                weights = self.calculate_bilinear_weights(q, neighbour_points)
                placed_value = self.calculate_weighted_value(img, neighbour_points, weights)
                scaled_img[point[0], point[1]] = np.round(placed_value)

        return scaled_img

class BicubicInterpolation(Interpolation):
    def __init__(self):
        super().__init__()

    # Cubic Interpolation Specific Functions
    def get_cubic_neighbours(self, point, h, w):
        """Get neighbors for cubic interpolation. Uses a 4x4 grid and handles out-of-bounds cases."""
        points = []
        point_y_int = int(np.floor(point[0]))
        point_x_int = int(np.floor(point[1]))

        # Generate 4x4 neighboring points for cubic interpolation
        for y in range(point_y_int - 1, point_y_int + 3):
            for x in range(point_x_int - 1, point_x_int + 3):
                # Ensure points are within bounds (handle out-of-bounds using replication)
                y_clamped = min(max(y, 0), h - 1)
                x_clamped = min(max(x, 0), w - 1)
                points.append((y_clamped, x_clamped))
        return points

    def calculate_cubic_weights(self, point, neighbour_points):
        """Calculate weights for cubic interpolation."""
        weights = []

        for neighbour in neighbour_points:
            weight_x = self.cubic_kernel(neighbour[1] - point[1])
            weight_y = self.cubic_kernel(neighbour[0] - point[0])
            weights.append(weight_x * weight_y)
        return weights

    def cubic_kernel(self, t):
        """Cubic convolution kernel for interpolation."""
        t = np.abs(t)
        if t <= 1:
            return 1.5 * t**3 - 2.5 * t**2 + 1
        elif t < 2:
            return -0.5 * t**3 + 2.5 * t**2 - 4 * t + 2
        return 0

    def interpolate(self, img, scale):
        h, w = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        scaled_h = int(h * scale)
        scaled_w = int(w * scale)

        scaled_img = np.zeros((scaled_h, scaled_w, channels), dtype=img.dtype)

        for j in range(scaled_h):
            for i in range(scaled_w):
                point = (j, i)
                q = self.get_reverse_scaled_point(point, scale, h, w)
                neighbour_points = self.get_cubic_neighbours(q, h, w)
                weights = self.calculate_cubic_weights(q, neighbour_points)
                placed_value = self.calculate_weighted_value(img, neighbour_points, weights)
                scaled_img[point[0], point[1]] = np.round(placed_value)

        return scaled_img

def read_image(filename, gray_scale = False):
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
    cv2.imwrite(output_folder+filename, img)

    
def rotate_upsample(img, scale, degree, interpolation_type):
    '''img: img to be rotated and upsampled
    scale: scale of upsampling (e.g. if current width and height is 64x64, and scale is 4, wodth and height of the output should be 256x256)
    degree: shows the degree of rotation
    interp: either linear or cubic'''
    rotation = Rotation()
    rotated_img = rotation.rotate_image(img, -degree)
    if interpolation_type == Interpolation.INTERPOLATION_TYPE.LINEAR:
        interpolation = BilinearInterpolation()
    elif interpolation_type == Interpolation.INTERPOLATION_TYPE.CUBIC:
        interpolation = BicubicInterpolation()
    img = interpolation.interpolate(rotated_img, scale)
    
    return img


def compute_histogram(image, bins=50):
    """Compute the normalized histogram of the hue channel."""
    # Convert image to HSI and extract the hue channel
    hsi_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel = hsi_img[:, :, 0]
    
    # Compute the histogram of the hue channel
    hist = cv2.calcHist([hue_channel], [0], None, [bins], [0, 180])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
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

    # Average the KL divergences for desert and forest images
    avg_kl_desert = (kl_desert1 + kl_desert2) / 2.0
    avg_kl_forest = (kl_forest1 + kl_forest2) / 2.0

    # Classify based on which average KL divergence is smaller
    if avg_kl_desert < avg_kl_forest:
        return 'desert'
    else:
        return 'forest'
    
def difference_images(img1, img2):
    '''img1 and img2 are the images to take dhe difference
    returns the masked image'''

    return masked_image

def main():
    def Q1():
        ###################### Q1
        # Read original image
        # Read corrupted image
        img_original = read_image('q1_1.png')
        img_path = 'ratio_4_degree_30.png'
        img = read_image(img_path)
        # Correct the image with linear interpolation
        corrected_img_linear = rotate_upsample(img, 4, 30, Interpolation.INTERPOLATION_TYPE.LINEAR)
        print("q1_1_corrected_linear.png is saved")
        write_image(corrected_img_linear, 'q1_1_corrected_linear.png')
        # Correct the image with cubic interpolation
        corrected_img_cubic = rotate_upsample(img, 4, 30, Interpolation.INTERPOLATION_TYPE.CUBIC)
        print("q1_1_corrected_cubic.png is saved")
        write_image(corrected_img_cubic, 'q1_1_corrected_cubic.png')

        # # Report the distances
        # print('The distance between original image and image corrected with linear interpolation is ', compute_distance(img_original, corrected_img_linear))
        # print('The distance between original image and image corrected with cubic interpolation is ', compute_distance(img_original, corrected_img_cubic))

        # Repeat the same steps for the second image
        img_original = read_image('q1_2.png')
        img = read_image('ratio_8_degree_45.png')
        corrected_img_linear = rotate_upsample(img, 8, 45, Interpolation.INTERPOLATION_TYPE.LINEAR)
        print("q1_2_corrected_linear.png is saved")
        write_image(corrected_img_linear, 'q1_2_corrected_linear.png')
        corrected_img_cubic = rotate_upsample(img, 8, 45, Interpolation.INTERPOLATION_TYPE.CUBIC)
        print("q1_2_corrected_cubic.png is saved")
        write_image(corrected_img_cubic, 'q1_2_corrected_cubic.png')

        # # Report the distances
        # print('The distance between original image and image corrected with linear interpolation is ', compute_distance(img_original, corrected_img_linear))
        # print('The distance between original image and image corrected with cubic interpolation is ', compute_distance(img_original, corrected_img_cubic))

    def Q2():
        # ###################### Q2
        img = read_image('q2_1.jpg')
        result = desert_or_forest(img)
        print("Given image q2_1 is an image of a ", result)

        img = read_image('q2_2.jpg')
        result = desert_or_forest(img)
        print("Given image q2_2 is an image of a ", result)

    def Q3():
        # ###################### Q3
        img1 = read_image('q3_a1.png',gray_scale=True)
        img2 = read_image('q3_a2.png',gray_scale=True)
        result = difference_images(img1,img2)
        write_image(result, 'masked_image_a.png')

        img1 = read_image('q3_b1.png')
        img2 = read_image('q3_b2.png')
        result = difference_images(img1,img2)
        write_image(result, 'masked_image_b.png')

    Q1()
def test():
    img = read_image('ratio_4_degree_30.png')
    # img = read_image('cat.jpg')
    cv2.imshow('img', img)
    cv2.waitKey(0)
    rotation = Rotation()
    rotated_img = rotation.rotate_image(img, -30)
    cv2.imshow('rotated_img', rotated_img)
    cv2.waitKey(0)
    biliear_interpolation = BilinearInterpolation()
    bilinear_interpolated_img = biliear_interpolation.interpolate(rotated_img, 4)
    cv2.imshow('bilinear_interpolated_img', bilinear_interpolated_img)
    cv2.waitKey(0)
    cubic_interpolation = CubicInterpolation()
    cubic_interpolated_img = cubic_interpolation.interpolate(rotated_img, 4)
    cv2.imshow('cubic_interpolated_img', cubic_interpolated_img)
    cv2.waitKey(0)




if __name__ == '__main__':
    main()
