from cv2 import dct
from cv2 import idct

from .block import Block


def encode(block: Block) -> Block:
    block.elements = dct(block.elements * 1.0)
    return block


def decode(block: Block) -> Block:
    block.elements = idct(block.elements * 1.0)
    return block
