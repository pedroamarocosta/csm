from numpy import int

from .dc import DC


def encode(previous_block, block) -> DC:
    if previous_block is None:
        return DC(block.elements[0, 0])

    return DC(block.elements[0, 0] - previous_block.elements[0, 0])


def decode(previous_block, dc: DC) -> int:
    if previous_block is None:
        return dc.elements

    return dc.elements + previous_block.elements[0, 0]
