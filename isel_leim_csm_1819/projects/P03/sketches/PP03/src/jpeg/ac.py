from numpy import abs
from numpy import binary_repr
from numpy import int
from numpy import ndarray, zeros
from numpy import sum


class AC:

    def __init__(self, zero_run_length: ndarray, amplitude: ndarray):
        self.__zero_run_lengths = zero_run_length
        self.__amplitudes = amplitude
        self.__sizes = zeros(self.__amplitudes.shape[0]).astype(int)

        for i in range(self.__amplitudes.shape[0]):
            bits = binary_repr(abs(self.__amplitudes[i]))
            self.__sizes[i] = int(len(bits))

    @property
    def zero_run_lengths(self):
        return self.__zero_run_lengths

    @property
    def amplitudes(self):
        return self.__amplitudes

    @property
    def elements(self):
        zeros_index = self.__zero_run_lengths > 0
        total_zeros = sum(self.__zero_run_lengths[zeros_index])

        elements = zeros(self.__amplitudes.shape[0] + total_zeros)

        i = 0
        j = 0

        while i < len(elements):
            if self.__zero_run_lengths[j] > 0:
                i += self.__zero_run_lengths[j]

            elements[i] = self.__amplitudes[j]

            j += 1
            i += 1

        return elements

    @property
    def sizes(self):
        return self.__sizes

    def __str__(self):
        text = ""

        for i in range(len(self.__amplitudes)):
            text = "{}{}({zrl}, {size})({amplitude})".format(
                text,
                " " if i == 0 else "",
                zrl=self.__zero_run_lengths[i],
                size=self.__sizes[i],
                amplitude=self.__amplitudes[i]
            )

        return text
