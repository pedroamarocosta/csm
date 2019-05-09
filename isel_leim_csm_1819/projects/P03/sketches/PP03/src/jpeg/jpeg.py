import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from .entropic_coding import Stream
from . import dct
from . import quantification
from . import ycc
from .block import Block


class JPEG:

    header_size = 16 * 2
    total_layers = 3

    def __init__(self, image: np.ndarray, quality: int = 50):
        self.__raw_image = np.copy(image)
        self.__image = np.copy(image)
        self.__quality = quality
        self.__stream = Stream()
        self.__header = Stream()

    @classmethod
    def from_stream(cls, stream: str):
        cls.__header, cls.__stream = cls.__decode_header(stream)

        return cls

    def __crop_image(self):
        total_lines, total_columns, layers = self.__image.shape

        excess_lines = total_lines % Block.size
        excess_columns = total_columns % Block.size

        for i in range(layers):
            self.__image[:, :, i] = self.__image[
                                    0: total_lines - excess_lines,
                                    0: total_columns - excess_columns, i
                                ]

    def __apply_subsample(self, image: np.array, horizontal_ratio: int, vertical_ratio: int, sample_size: int = 4):
        rows, columns = image.shape
        x = 0
        y = 0
        for row in range(rows):
            for col in range(columns):
                if x < rows and y < columns:
                    sub_matrix = np.ones((1, horizontal_ratio, vertical_ratio)) * image[x][y]
                    image = self.__apply_block(image, sub_matrix, x, y, vertical_ratio)
                    y += vertical_ratio
            x += horizontal_ratio
            y = 0

        return image

    def __apply_block(self, matrix, block, line, col, dim):
        matrix[line: line + dim, col: col + dim] = block
        return matrix

    def __make_subsample(self, horizontal_ratio: int, vertical_ratio: int):
        for frame in range(1, 3):
            self.__image[:, :, frame] = self.__apply_subsample(self.__image[:, :, frame], horizontal_ratio, vertical_ratio)

    def encode(self, horizontal_ratio: int, vertical_ratio: int):
        self.__crop_image()
        if horizontal_ratio != 1:
            self.__make_subsample(horizontal_ratio, vertical_ratio)
        self.__image = ycc.encode(self.__image)

        for i in range(JPEG.total_layers):
            blocks = self.__make_blocks(self.__image[:, :, i])
            previous_block = None

            if i == 0:
                self.__encode_header(blocks.shape[0], blocks.shape[1])

            for row in range(blocks.shape[0]):
                for column in range(blocks.shape[1]):
                    block = dct.encode(blocks[row, column])
                    block = quantification.encode(blocks[row, column], self.__quality)

                    block.encode(previous_block)
                    self.__stream.join(block.stream)

                    previous_block = copy.deepcopy(block)

    def decode(self):

        if self.__header == "":
            return

        if self.__stream == "":
            return

        stream = copy.deepcopy(self.__stream)
        rows = int(self.__header[0: int(JPEG.header_size / 2)], 2)
        columns = int(self.__header[int(JPEG.header_size / 2): JPEG.header_size], 2)

        image = np.zeros(
            (columns * Block.size, rows * Block.size, JPEG.total_layers),
            dtype=np.uint8
        )

        for i in range(JPEG.total_layers):
            previous_block = None

            for row in range(rows):
                for column in range(columns):
                    block, stream = Block.decode(previous_block, stream)
                    previous_block = copy.deepcopy(block)

                    block = quantification.decode(block)
                    block = dct.decode(block)

                    image[
                        row * Block.size: row * Block.size + Block.size,
                        column * Block.size: column * Block.size + Block.size,
                        i,
                    ] = block.elements

        image = ycc.decode(image)
        self.__image = image

    def __encode_header(self, rows: int, columns: int):
        row_bits = np.binary_repr(rows).rjust(np.int(JPEG.header_size / 2), "0")
        column_bits = np.binary_repr(columns).rjust(np.int(JPEG.header_size / 2), "0")

        self.__header.add_prefix(row_bits)
        self.__header.add_prefix(column_bits)

    @staticmethod
    def __decode_header(stream: str) -> (Stream, int, int, str):
        header = Stream()
        header.add_prefix(stream[0: np.int(JPEG.header_size)])

        stream = stream[JPEG.header_size:]

        return header, stream

    def __subtract_128(self):
        self.__image = self.__image - 128

    def __add_128(self):
        self.__image = self.__image + 128

    def __make_blocks(self, image_layer: np.ndarray):
        # Calculate the total number of blocks
        rows = image_layer.shape[0]
        columns = image_layer.shape[1]

        total_blocks_horizontal = np.int(columns / Block.size)
        total_blocks_vertical = np.int(rows / Block.size)
        total_blocks = total_blocks_horizontal * total_blocks_vertical

        # Initialize a bi-dimensional array of blocks
        blocks = [Block] * total_blocks
        blocks = np.reshape(blocks, (total_blocks_horizontal, total_blocks_vertical))

        # Make each block
        for blocks_row in range(blocks.shape[0]):
            for blocks_column in range(blocks.shape[1]):
                elements_row = blocks_row * Block.size
                elements_column = blocks_column * Block.size
                elements = image_layer[
                               elements_row: elements_row + Block.size,
                               elements_column: elements_column + Block.size
                           ]

                blocks[blocks_row, blocks_column] = Block(elements)

        return blocks

    @property
    def stream(self):
        if self.__header is None or self.__stream is None:
            self.encode()

        final_stream = Stream()
        final_stream.join(self.__header).join(self.__stream)

        return final_stream.regular

    @property
    def image(self):
        return self.__image
