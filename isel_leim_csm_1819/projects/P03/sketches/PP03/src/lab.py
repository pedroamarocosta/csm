from cv2 import imread, imwrite
import cv2
from time import clock
from numpy import ndarray
import matplotlib.pyplot as plt
from jpeg import JPEG
from jpeg import snr
from jpeg import comp_rate


def process(image: ndarray, horizontal_ratio: int, vertical_ratio: int, quality: int = 50):

    print("Encoding image with JPEG")
    t = clock()

    image = JPEG(image, quality)
    image.encode(horizontal_ratio, vertical_ratio)

    elapsed_time = clock() - t
    print("Encoding time: {}".format(round(elapsed_time, 2)) + " seconds")

    print("Decoding stream")
    t = clock()

    image.decode()

    elapsed_time = clock() - t
    print("Decoding time: {}".format(round(elapsed_time, 2)) + " seconds")

    return image


def write_to_file(stream: str, filename: str):
    with open(filename, "w") as file:
        file.write(stream)

