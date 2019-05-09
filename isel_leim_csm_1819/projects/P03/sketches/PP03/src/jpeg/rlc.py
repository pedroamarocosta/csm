from numpy import argsort
from numpy import argwhere
from numpy import array, ndarray, zeros
from numpy import int, uint
from numpy import sum

from .ac import AC

zigzag = array([
    [0, 1, 5, 6, 14, 15, 27, 28],
    [2, 4, 7, 13, 16, 26, 29, 42],
    [3, 8, 12, 17, 25, 30, 41, 43],
    [9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63],
])

original_index = zigzag.reshape(64, order="F").astype("int")
zigzag_index = argsort(original_index)


def encode(block) -> AC:
    elements = block.elements.flatten(order="F")[zigzag_index]

    idx = elements != 0
    amplitudes = elements[idx][1:]
    zrls = zeros(amplitudes.shape[0], dtype=uint)

    j = 1
    idx = argwhere(idx).flatten()[1:]

    for i in range(len(zrls)):
        total_zeros = sum(elements[j: idx[i]] == 0)

        # TODO: What happens when there is more then 15 zeros in ZRL?
        if total_zeros > 0:
            zrls[i] = total_zeros

        # if total_zeros > 15:
        #     print("[WARNING] ZRL higher then 15, ({}, {})".format(total_zeros, abs(amplitudes[i])))

        j = idx[i]

    return AC(zrls, amplitudes)


def decode(ac: AC, total_elements: int) -> ndarray:
    elements = zeros(total_elements, dtype=int)

    zrls = ac.zero_run_lengths
    amplitudes = ac.amplitudes

    # The first element is the DC part
    j = 1

    for i in range(len(amplitudes)):
        if zrls[i] > 0:
            j += zrls[i]

        elements[j] = amplitudes[i]
        j += 1

    elements = elements[original_index].reshape(
        zigzag.shape[0], zigzag.shape[1],
        order="F"
    )

    return elements

