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


def bilinear_scale(original_img, new_h, new_w):
	#get dimensions of original image
	old_h, old_w, c = original_img.shape
	#create an array of the desired shape.
	#We will fill-in the values later.
	resized = np.zeros((new_h, new_w, c))
	#Calculate horizontal and vertical scaling factor
	w_scale_factor = (old_w ) / (new_w ) if new_h != 0 else 0
	h_scale_factor = (old_h ) / (new_h ) if new_w != 0 else 0
	for i in range(new_h):
		for j in range(new_w):
			#map the coordinates back to the original image
			x = i * h_scale_factor
			y = j * w_scale_factor
			#calculate the coordinate values for 4 surrounding pixels.
			x_floor = math.floor(x)
			x_ceil = min( old_h - 1, math.ceil(x))
			y_floor = math.floor(y)
			y_ceil = min(old_w - 1, math.ceil(y))

			if (x_ceil == x_floor) and (y_ceil == y_floor):
				q = original_img[int(x), int(y), :]
			elif (x_ceil == x_floor):
				q1 = original_img[int(x), int(y_floor), :]
				q2 = original_img[int(x), int(y_ceil), :]
				q = q1 * (y_ceil - y) + q2 * (y - y_floor)
			elif (y_ceil == y_floor):
				q1 = original_img[int(x_floor), int(y), :]
				q2 = original_img[int(x_ceil), int(y), :]
				q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor))
			else:
				v1 = original_img[x_floor, y_floor, :]
				v2 = original_img[x_ceil, y_floor, :]
				v3 = original_img[x_floor, y_ceil, :]
				v4 = original_img[x_ceil, y_ceil, :]

				q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
				q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
				q = q1 * (y_ceil - y) + q2 * (y - y_floor)

			resized[i,j,:] = q
	return resized

# Interpolation kernel
def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1

    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s) ** 3) - (5 * a) * (abs(s) ** 2) + (8 * a) * abs(s) - 4 * a
    return 0

# Padding
def padding(img, H, W, C):
    zimg = np.zeros((H + 4, W + 4, C))
    zimg[2:H + 2, 2:W + 2, :C] = img

    # Pad the first/last two col and row
    zimg[2:H + 2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H + 2:H + 4, 2:W + 2, :] = img[H - 1:H, :, :]
    zimg[2:H + 2, W + 2:W + 4, :] = img[:, W - 1:W, :]
    zimg[0:2, 2:W + 2, :C] = img[0:1, :, :C]

    # Pad the missing eight points
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H + 2:H + 4, 0:2, :C] = img[H - 1, 0, :C]
    zimg[H + 2:H + 4, W + 2:W + 4, :C] = img[H - 1, W - 1, :C]
    zimg[0:2, W + 2:W + 4, :C] = img[0, W - 1, :C]

    return zimg


# Bicubic operation
def bicubic_interpolation(img, scale, a):
    # Get image size
    H, W, C = img.shape

    # Here H = Height, W = weight,
    # C = Number of channels if the
    # image is coloured.
    img = padding(img, H, W, C)

    # Create new image
    dH = math.floor(H * scale)
    dW = math.floor(W * scale)

    # Converting into matrix
    dst = np.zeros((dH, dW, 3))

    # np.zeroes generates a matrix
    # consisting only of zeroes
    # Here we initialize our answer
    # (dst) as zero

    h = 1 / scale

    print('Start bicubic interpolation')
    print('It will take a little while...')
    inc = 0

    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                # Getting the coordinates of the
                # nearby values
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                # Considering all nearby 16 values
                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[img[int(y - y1), int(x - x1), c],
                                    img[int(y - y2), int(x - x1), c],
                                    img[int(y + y3), int(x - x1), c],
                                    img[int(y + y4), int(x - x1), c]],
                                   [img[int(y - y1), int(x - x2), c],
                                    img[int(y - y2), int(x - x2), c],
                                    img[int(y + y3), int(x - x2), c],
                                    img[int(y + y4), int(x - x2), c]],
                                   [img[int(y - y1), int(x + x3), c],
                                    img[int(y - y2), int(x + x3), c],
                                    img[int(y + y3), int(x + x3), c],
                                    img[int(y + y4), int(x + x3), c]],
                                   [img[int(y - y1), int(x + x4), c],
                                    img[int(y - y2), int(x + x4), c],
                                    img[int(y + y3), int(x + x4), c],
                                    img[int(y + y4), int(x + x4), c]]])
                mat_r = np.matrix(
                    [[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])

                # Here the dot function is used to get the dot
                # product of 2 matrices
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)
    return dst


def rotate_upsample(img, scale, degree, interpolation_type):
    # Get image dimensions
    height = img.shape[0]
    width = img.shape[1]

    # Calculate center of the image (integer division)
    cx = width // 2
    cy = height // 2

    # Compute new dimensions
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Create an empty image for the rotated result
    rotated_img = np.zeros_like(img)
    scaled_img = np.zeros((new_height, new_width, 3))

    # Convert the angle to radians
    alpha = np.radians(degree)

    # Inverse rotation matrix (backward transformation)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    # Convert interpolation type to OpenCV flag
    if interpolation_type == 'linear':
        rotated_img = bilinear_rotate(img, degree)
        scaled_img = cv2.resize(rotated_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    elif interpolation_type == 'cubic':
        print("hello")
        #for y in range(height):
            #for x in range(width):
                # Translate coordinates to center
                #v, w = x - cx, y - cy

                # Apply backward rotation
                #x_prime = v * cos_alpha + w * sin_alpha + cx
                #y_prime = -v * sin_alpha + w * cos_alpha + cy

                # Get pixel value using bicubic interpolation
                #if 0 <= x_prime < width and 0 <= y_prime < height:
                    #img[x] = x_prime
                    #img[y] = y_prime
                    #rotated_img[y, x] = bicubic_interpolation(img, scale, -0.5)
        # Resize the image using bicubic interpolation
        #scaled_img = cv2.resize(rotated_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        #scaled_img = rotated_img

    else:
        raise ValueError("Invalid interpolation type")

    return scaled_img

def compute_distance(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape.")

    distance = np.mean((img1 - img2) ** 2)
    return distance

def hue_calc(img):
    # divide R G B channels
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    hue_img = np.copy(red)
    height = img[0]
    width = img[1]

    for j in range(height):
        for i in range(width):
            hue_img[j][i] = (0.5*(red[j][i]-green[j][i] + red[j][i]-blue[j][i])) / \
            (math.sqrt((red[j][i]-green[j][i])**2 + (red[j][i]-blue[j][i])*(green[j][i]-blue[j][i])))

            hue_img[j][i] = math.acos(hue_img[j][i])
    return hue_img

def histogram_calc(hue):
    height = hue[0]
    width = hue[1]
    histogram_img = np.zeros(height,width)

    for x in range(height):
        for y in range(width):
            histogram_img[hue[x,y]] += 1
    return histogram_img

def kl_divergence(p, q):
    print("effef")

def desert_or_forest(img):
    '''img: image to be classified as desert or forest
    return a string: either 'desert'  or 'forest'

    You should compare the KL Divergence between histograms of hue channel. Please provide images and discuss these histograms in your report'''
    desert1 = read_image('desert1.jpg')
    desert2 = read_image('desert2.jpg')
    forest1 = read_image('forest1.jpg')
    forest2 = read_image('forest2.jpg')


def difference_images(img1, img2, threshold=30):
    """
    Masks out the differences between img1 and img2 and applies the mask to img2.
    Assumes that an object is absent in img1 and present in img2.

    :param img1: The first image (before the object appears)
    :param img2: The second image (after the object appears)
    :param threshold: The threshold value for filtering small differences (default 30)
    :return: The image2 with the mask applied
    """

    # Convert images to grayscale for easier comparison
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the two grayscale images
    diff = cv2.absdiff(gray2, gray1)

    # Apply a threshold to filter out small differences (e.g., lighting changes, noise)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Optionally perform morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Convert the mask to 3 channels to apply it on the color image
    mask_color = cv2.merge([mask, mask, mask])

    # Apply the mask to img2 (highlighting the new object)
    result = cv2.bitwise_and(img2, mask_color)

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




