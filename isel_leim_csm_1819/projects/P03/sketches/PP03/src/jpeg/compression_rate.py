import os


def calc_compression_rate(original_file, encoded_file):
    original_size = os.path.getsize(original_file)
    encoded_size = os.path.getsize(encoded_file)
    rate = round(100 * (1 - encoded_size / original_size), 2)
    ratio = round(1. * original_size / encoded_size, 2)

    return ratio, rate
