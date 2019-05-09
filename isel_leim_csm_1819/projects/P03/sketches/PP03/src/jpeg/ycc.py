from cv2 import COLOR_BGR2YCrCb
from cv2 import COLOR_YCrCb2BGR
from cv2 import cvtColor

from numpy import array


def encode(image: array) -> array:
    return cvtColor(image, COLOR_BGR2YCrCb)


def decode(image: array) -> array:
    return cvtColor(image, COLOR_YCrCb2BGR)
