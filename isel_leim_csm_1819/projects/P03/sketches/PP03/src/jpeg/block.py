from numpy import copy
from numpy import ndarray

from . import dpcm
from . import entropic_coding
from . import rlc
from .entropic_coding import Stream


class Block:

    size = 8

    def __init__(self, elements: ndarray):
        self.__elements = copy(elements)
        self.__dc = None
        self.__ac = None
        self.__stream = Stream()

    def __getitem__(self, item):
        return self.__elements[item]

    def __setitem__(self, key, value):
        self.__elements[key] = value

    def __str__(self):
        return self.__elements.__str__()

    @property
    def elements(self):
        return copy(self.__elements)

    @elements.setter
    def elements(self, value):
        self.__elements = value

    @property
    def dc(self):
        return self.__dc

    @dc.setter
    def dc(self, value):
        self.__dc = value

    @property
    def ac(self):
        return self.__ac

    @ac.setter
    def ac(self, value):
        self.__ac = value

    @property
    def stream(self):
        return self.__stream

    def encode(self, previous_block) -> Stream:
        self.__dc = dpcm.encode(previous_block, self)
        self.__ac = rlc.encode(self)
        self.__stream = entropic_coding.encode(self.__dc, self.__ac)

        return self.__stream

    @staticmethod
    def decode(previous_block, stream: Stream):
        dc, ac, stream = entropic_coding.decode(stream, Block.size * Block.size)

        if dc is None or ac is None:
            return None

        elements = rlc.decode(ac, Block.size * Block.size)
        elements[0, 0] = dpcm.decode(previous_block, dc)

        block = Block(elements)

        block.dc = dc
        block.ac = ac

        return block, stream
