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
    if gray_scale:
        img = cv2.imread(input_folder + filename, cv2.IMREAD_GRAYSCALE)
        return img
    img = cv2.imread(input_folder + filename)
    return img


def write_image(img, filename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(output_folder + filename, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def apply_compression(image_path, output_prefix, n_values):
    # Read grayscale image
    image = read_image(image_path, gray_scale=True)
    write_image(image, f"{output_prefix}_original.jpg")
    original_size = os.path.getsize(output_folder + f"{output_prefix}_original.jpg") / 1024
    rows, cols = image.shape

    results = {}

    # Store the original size and resolution
    results["Size Original:"] = original_size
    results["JPEG Quality"] = "90%"
    results["n_results"] = {}

    # Haar Wavelet Transform
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Discrete Cosine Transform (DCT)
    image_dct = dct(dct(image.T, norm='ortho').T, norm='ortho')

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

        # Calculate Normalized MSE
        normalized_mse_haar = (mse_haar / (255 ** 2)) * 100
        normalized_mse_dct = (mse_dct / (255 ** 2)) * 100

        # Calculate PSNR
        psnr_haar = 10 * np.log10((255 ** 2) / mse_haar) if mse_haar > 0 else float('inf')
        psnr_dct = 10 * np.log10((255 ** 2) / mse_dct) if mse_dct > 0 else float('inf')

        # Store results for this N
        results["n_results"][n] = {
            "Normalized MSE Haar (%)": normalized_mse_haar,
            "Normalized MSE DCT (%)": normalized_mse_dct,
            "PSNR Haar (dB)": psnr_haar,
            "PSNR DCT (dB)": psnr_dct
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
def print_results(results):
    for index, result in enumerate(results):
        print("-----------------")
        print(f"c{index+1}.jpg")
        print("-----------------")
        for k, v in result.items():
            if k != "n_results":
                print(f"{k}: {v}")
        print("---")
        for n, metrics in result["n_results"].items():
            print(f"N = {n}%")
            for k, v in metrics.items():
                print(f"{k}: {v}")
            print("---")

print_results([results1, results2, results3])
