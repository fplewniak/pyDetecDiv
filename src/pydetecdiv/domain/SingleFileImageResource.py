"""
 A class handling image resource in a single 5D TIFF file
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydetecdiv.domain.ImageResource import ImageResource

import psutil
from bioio import BioImage
from bioio_base.dimensions import Dimensions
from tifffile import tifffile
import numpy as np
import cv2

from pydetecdiv.domain.ImageResourceData import ImageResourceData
from pydetecdiv.domain.Image import Image, ImgDType


class SingleFileImageResource(ImageResourceData):
    """
    A business-logic class defining valid operations and attributes of Image resources stored in a single 5D file
    """

    def __init__(self, image_resource: 'ImageResource' = None, max_mem: int = 5000, **kwargs):
        self.path = image_resource.image_files[0]
        self.fov = image_resource.fov
        self.image_resource = image_resource.id_
        self.max_mem = max_mem
        self._memmap = tifffile.memmap(self.path, **kwargs)
        # self.img_reader = AICSImage(self.path).reader
        self.img_reader = BioImage(self.path)
        self._drift = image_resource.drift

        # print(f'Single file image resource: {self.dims}')

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The image resource shape (should be 5D with the following dimensions TCZYX)
        """
        return self.img_reader.shape

    @property
    def dims(self) -> Dimensions:
        """
        the image dimensions
        """
        return self.img_reader.dims

    @property
    def sizeT(self) -> int:
        """
        The number of frames
        """
        return self.img_reader.dims.T

    @property
    def sizeC(self) -> int:
        """
        The number of channels
        """
        return self.img_reader.dims.C

    @property
    def sizeZ(self) -> int:
        """
        The number of layers
        """
        return self.img_reader.dims.Z

    @property
    def sizeY(self) -> int:
        """
        The height of image
        """
        return self.img_reader.dims.Y

    @property
    def sizeX(self) -> int:
        """
        The width of image
        """
        return self.img_reader.dims.X

    def _image(self, C: int = 0, Z: int = 0, T: int = 0, drift: bool = False) -> np.ndarray:
        """
        A 2D grayscale image (on frame, one channel and one layer)

        :param C: the channel index
        :param Z: the layer index
        :param T: the frame index
        :param drift: True if the drift correction should be applied
        :return: a 2D data array
        """
        s = self.shape
        data = np.expand_dims(self._memmap, axis=tuple(i for i in range(len(s)) if s[i] == 1))[T, C, Z, ...]

        if drift and self.drift is not None:
            data = cv2.warpAffine(np.array(data),
                                  np.float32(
                                          [[1, 0, -self.drift.iloc[T].dx],
                                           [0, 1, -self.drift.iloc[T].dy]]),
                                  (data.shape[1], data.shape[0]))
        # data = tf.image.convert_image_dtype(data, dtype=tf.uint16, saturate=False).numpy()
        data = Image(data).as_array(dtype=ImgDType.uint16)
        return data

    def _image_memmap(self, sliceX: slice = None, sliceY: slice = None, C: int = 0, Z: int = 0, T: int = 0,
                      drift: bool = False) -> np.ndarray:
        if sliceX is None:
            sliceX = slice(0, self.sizeX)
        if sliceY is None:
            sliceY = slice(0, self.sizeX)
        deltaX = 0 if not drift or self.drift is None else int(round(self.drift.iloc[T].dx))
        deltaY = 0 if not drift or self.drift is None else int(round(self.drift.iloc[T].dy))

        sliceX = slice(sliceX.start - deltaX, sliceX.stop - deltaX)
        sliceY = slice(sliceY.start - deltaY, sliceY.stop - deltaY)

        s = self.shape
        data = np.expand_dims(self._memmap, axis=tuple(i for i in range(len(s)) if s[i] == 1))[T, C, Z, sliceY, sliceX]
        return data

    def data_sample(self, X: slice = None, Y: slice = None) -> np.ndarray:
        """
        Return a sample from an image resource, specified by X and Y slices. This is useful to extract resources for
        regions of interest from a field of view.

        :param X: the X slice
        :param Y: the Y slice
        :return: the sample data (in-memory)
        """
        s = self.shape
        data = np.expand_dims(self._memmap, axis=tuple(i for i in range(len(s)) if s[i] == 1))[..., Y, X]
        return data

    def open(self) -> None:
        """
        Open the memory mapped file to access data
        """
        self._memmap = tifffile.memmap(self.path)

    def close(self) -> None:
        """
        Close the memory mapped file
        """
        if not self._memmap._mmap.closed:
            self._memmap._mmap.close()

    def flush(self) -> None:
        """
        Flush the data to save changes to the memory mapped file
        """
        if not self._memmap._mmap.closed:
            self._memmap._mmap.flush()

    def refresh(self) -> None:
        """
        Close and open the memory mapped file to save memory if needed. Useful when creating a new file or making lots
        of changes.
        """
        if psutil.Process().memory_info().rss / (1024 * 1024) > self.max_mem:
            self.close()
            self.open()
