import numpy as np
import cv2
from scipy.ndimage import rotate

def cubic_kernel(t):
    """Cubic convolution kernel for interpolation."""
    t = np.abs(t)
    if t <= 1:
        return 1.5 * t**3 - 2.5 * t**2 + 1
    elif t < 2:
        return -0.5 * t**3 + 2.5 * t**2 - 4 * t + 2
    return 0

def interpolate_1d(pixels, t):
    """Perform 1D interpolation using the cubic kernel."""
    result = 0.0
    for i in range(-1, 3):  # Using 4 pixels in the interpolation
        result += pixels[i + 1] * cubic_kernel(i - t)
    return result

def get_pixel(image, x, y, border_type="replicate"):
    """Handle out-of-bounds access using replicate border replication."""
    height, width = image.shape[:2]

    if border_type == "replicate":
        # Replicate edge pixels
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
    
    return image[int(y), int(x)]

def bicubic_interpolation(image, x, y):
    """Perform bicubic interpolation on a 2D image or 3D image (multi-channel)."""
    if image.ndim == 2:  # Grayscale image
        return _bicubic_interpolation_single_channel(image, x, y)
    elif image.ndim == 3:  # Multi-channel (RGB) image
        channels = []
        for channel in range(image.shape[2]):
            channel_data = _bicubic_interpolation_single_channel(image[:, :, channel], x, y)
            channels.append(channel_data)
        return np.stack(channels, axis=-1)

def _bicubic_interpolation_single_channel(image, x, y):
    """Bicubic interpolation for a single channel."""
    x_floor = int(np.floor(x))
    y_floor = int(np.floor(y))

    dx = x - x_floor
    dy = y - y_floor

    result = 0.0
    for j in range(-1, 3):  # Iterate over 4 neighboring rows
        row_interpolation = 0.0
        for i in range(-1, 3):  # Iterate over 4 neighboring columns
            pixel_value = get_pixel(image, x_floor + i, y_floor + j, border_type="replicate")
            row_interpolation += pixel_value * cubic_kernel(i - dx)
        
        result += row_interpolation * cubic_kernel(j - dy)
    
    return result

def resize_bicubic(image, scale_x, scale_y):
    """Resize an image using bicubic interpolation."""
    height, width = image.shape[:2]
    if image.ndim == 3:
        channels = image.shape[2]
        new_image = np.zeros((int(height * scale_y), int(width * scale_x), channels), dtype=np.float32)
    else:
        new_image = np.zeros((int(height * scale_y), int(width * scale_x)), dtype=np.float32)

    for j in range(int(height * scale_y)):
        for i in range(int(width * scale_x)):
            x = i / scale_x
            y = j / scale_y
            new_image[j, i] = bicubic_interpolation(image, x, y)
    
    return new_image

# Sample image loading and rotation using SciPy
img = cv2.imread('THE1_Images/ratio_4_degree_30.png')
if img is None:
    raise FileNotFoundError("Image file not found.")
img = np.array(img, dtype=np.float32)
img = rotate(img, -30, reshape=True)

# Rescale image by a factor of 2 using replicate border
scaled_image = resize_bicubic(img, 4, 4)

# Display results
cv2.imshow('Original Image', img / 255.0)  # Normalize for display
cv2.waitKey(0)
cv2.imshow('Scaled Image', scaled_image / 255.0)
cv2.waitKey(0)
cv2.destroyAllWindows()
