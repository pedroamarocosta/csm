from numpy import abs
from numpy import int
from numpy import binary_repr


class DC:

    def __init__(self, amplitude: int, size: int=0):
        self.__amplitude = amplitude
        self.__size = size

        if self.__amplitude != 0:
            self.__size = len(binary_repr(abs(self.__amplitude)))

    @property
    def amplitude(self):
        return self.__amplitude

    @property
    def size(self):
        return self.__size

    @property
    def elements(self):
        return self.__amplitude

    def __str__(self):
        return "({size}, {amplitude})".format(
            size=self.__size,
            amplitude=self.__amplitude
        )
