import numpy as np

from .block import Block


def quality_factor(q: float) -> float:

    if q >= 100:
        return 1

    if q <= 50:
        return 50 / q

    return 2 - (q * 2) / 100


# k1
def get_matrix() -> np.array:
    return np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])


def encode(block: Block, quality: float=50) -> Block:
    k = get_matrix()
    block.elements = np.round(block.elements / (k * quality_factor(quality))).astype(int)
    return block


def decode(block: Block, quality: float=50) -> Block:
    k = get_matrix()
    block.elements = block.elements * (k * quality_factor(quality))
    return block

