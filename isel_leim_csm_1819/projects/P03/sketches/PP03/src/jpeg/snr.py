import numpy as np


def calc_snr(original_image: np.ndarray, encoded_image: np.ndarray) -> float:
    original_image = original_image.flatten()
    encoded_image = encoded_image.flatten()

    error = original_image - encoded_image

    original_power = np.sum(original_image ** 2.0) / len(original_image)
    error_power = np.sum(error ** 2.0) / len(error)
    snr = 10 * np.log10(original_power / error_power)

    return round(snr, 2)


def calc_psnr(original_image: np.ndarray, encoded_image: np.ndarray):

    max_value = np.power(np.max(original_image), 2)
    mean_squared_error = (1 / (3 * encoded_image.shape[0] * encoded_image.shape[1])) * \
                         np.sum(np.power((original_image - encoded_image), 2))
    psnr = 10 * np.log10(max_value / mean_squared_error)
    return round(psnr, 2)
