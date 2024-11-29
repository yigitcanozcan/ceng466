# CENG 466 THE2
# Mert Uludoğan 2380996
# Yiğitcan Özcan 2521847

#ISSUE: When N gets bigger the compressed image's size surpasses the original image. WTF?

import numpy as np
import cv2
import os
import pywt
from scipy.fftpack import dct, idct
from skimage.metrics import mean_squared_error

### COMMON FUNCTIONS ###
input_folder = 'THE2_Images/Question3/'
output_folder = 'THE2_Images/Question3/'


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



def apply_compression(image_path, output_prefix, n_values):
    # Read grayscale image
    image = read_image(image_path, gray_scale=True)
    rows, cols = image.shape

    # Haar Wavelet Transform
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Discrete Cosine Transform (DCT)
    image_dct = dct(dct(image.T, norm='ortho').T, norm='ortho')

    results = {}

    for n in n_values:
        # Haar Wavelet Compression
        haar_coeffs = compress_wavelet((cA, (cH, cV, cD)), n)
        compressed_haar = pywt.idwt2(haar_coeffs, 'haar')

        # Discrete Cosine Transform Compression
        dct_compressed = compress_dct(image_dct, n)
        reconstructed_dct = idct(idct(dct_compressed.T, norm='ortho').T, norm='ortho')

        # Compute MSE
        mse_haar = mean_squared_error(image, compressed_haar)
        mse_dct = mean_squared_error(image, reconstructed_dct)

        # Save compressed images
        write_image(compressed_haar, f"{output_prefix}_haar_{n}.jpg")
        write_image(reconstructed_dct, f"{output_prefix}_dct_{n}.jpg")

        # Store results
        results[n] = {
            "mse_haar": mse_haar,
            "mse_dct": mse_dct,
            "haar_image": compressed_haar,
            "dct_image": reconstructed_dct
        }

    return results

def compress_wavelet(coeffs, N):
    """Compress wavelet coefficients by retaining top N%."""
    cA, (cH, cV, cD) = coeffs
    coeffs_flat = np.hstack([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
    threshold = np.percentile(np.abs(coeffs_flat), 100 - N)
    cA = np.where(np.abs(cA) < threshold, 0, cA)
    cH = np.where(np.abs(cH) < threshold, 0, cH)
    cV = np.where(np.abs(cV) < threshold, 0, cV)
    cD = np.where(np.abs(cD) < threshold, 0, cD)
    return cA, (cH, cV, cD)

def compress_dct(dct_matrix, N):
    """Compress DCT coefficients by retaining top N%."""
    coeffs_flat = dct_matrix.flatten()
    threshold = np.percentile(np.abs(coeffs_flat), 100 - N)
    compressed_dct = np.where(np.abs(dct_matrix) < threshold, 0, dct_matrix)
    return compressed_dct


results1 = apply_compression("c1.jpg", "c1", [1, 10, 50])
results2 = apply_compression("c2.jpg", "c2", [1, 10, 50])
results3 = apply_compression("c3.jpg", "c3", [1, 10, 50])

# Analyze results
print("-----------------")
print("c1.jpg")
print("-----------------")
for n, metrics in results1.items():
    print(f"N = {n}%")
    print(f"MSE (Haar): {metrics['mse_haar']:.4f}")
    print(f"MSE (DCT): {metrics['mse_dct']:.4f}")
print("-----------------")
print("c2.jpg")
print("-----------------")
for n, metrics in results2.items():
    print(f"N = {n}%")
    print(f"MSE (Haar): {metrics['mse_haar']:.4f}")
    print(f"MSE (DCT): {metrics['mse_dct']:.4f}")
print("-----------------")
print("c3.jpg")
print("-----------------")
for n, metrics in results3.items():
    print(f"N = {n}%")
    print(f"MSE (Haar): {metrics['mse_haar']:.4f}")
    print(f"MSE (DCT): {metrics['mse_dct']:.4f}")