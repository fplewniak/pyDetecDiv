"""
 A class handling image resource in HDF5 file
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydetecdiv.domain.ImageResource import ImageResource

import os

import tensorflow as tf
import h5py
import numpy as np
import cv2
from bioio_base.dimensions import Dimensions
from pydetecdiv.domain.ImageResourceData import ImageResourceData
from pydetecdiv.settings import get_config_value


class Hdf5ImageResource(ImageResourceData):
    """
    A business-logic class defining valid operations and attributes of Image resources stored in HDF5 file
    """

    def __init__(self, max_mem: int = 5000, image_resource: 'ImageResource' = None):
        print('HDF5')
        self.path = os.path.join(get_config_value('project', 'workspace'),
                                 image_resource.fov.project.dbname, 'data',
                                 image_resource.key_val['hdf5'])
        self.fov = image_resource.fov
        self.image_resource = image_resource.id_
        self.max_mem = max_mem
        self._dims = image_resource.dims
        self._drift = image_resource.drift

        # print(f'Multiple file image resource: {self.dims}')

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The image resource shape (should be 5D with the following dimensions TCZYX)
        """
        with h5py.File(self.path, 'r') as hdf5_file:
            return hdf5_file[self.fov.name].shape

    @property
    def dims(self) -> Dimensions:
        """
        the dimensions of the image resource
        :return:
        """
        return Dimensions("TCZYX", self.shape)
        # return self._dims

    @property
    def sizeT(self) -> int:
        """
        The number of frames
        """
        return self.shape[0]
        # return self._dims.T

    @property
    def sizeC(self) -> int:
        """
        The number of channels
        """
        return self.shape[1]
        # return self._dims.C

    @property
    def sizeZ(self) -> int:
        """
        The number of layers
        """
        return self.shape[2]
        # return self._dims.Z

    @property
    def sizeY(self) -> int:
        """
        The height of image
        """
        return self.shape[3]
        # return self._dims.Y

    @property
    def sizeX(self) -> int:
        """
        The width of image
        """
        return self.shape[4]
        # return self._dims.X

    def _image(self, C: int = 0, Z: int = 0, T: int = 0, drift: bool = False) -> tf.Tensor:
        """
        A 2D grayscale image (on frame, one channel and one layer)

        :param C: the channel index
        :type C: int
        :param Z: the layer index
        :type Z: int
        :param T: the frame index
        :type T: int
        :param drift: True if the drift correction should be applied
        :type drift: bool
        :return: a 2D data array
        :rtype: 2D numpy.array
        """
        with h5py.File(self.path, 'r') as hdf5_file:
            data = hdf5_file[self.fov.name][T, C, Z]
            if drift and self.drift is not None:
                data = cv2.warpAffine(np.array(data),
                                      np.float32(
                                              [[1, 0, -self.drift.iloc[T].dx],
                                               [0, 1, -self.drift.iloc[T].dy]]),
                                      (data.shape[1], data.shape[0]))
            data = tf.image.convert_image_dtype(data, dtype=tf.uint16, saturate=False).numpy()
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

        with h5py.File(self.path, 'r') as hdf5_file:
            return hdf5_file[self.fov.name][T, C, Z, sliceY, sliceX]

    def data_sample(self, X: slice = None, Y: slice = None) -> np.ndarray:
        """
        Return a sample from an image resource, specified by X and Y slices. This is useful to extract resources for
        regions of interest from a field of view.

        :param X: the X slice
        :type X: slice
        :param Y: the Y slice
        :type Y: slice
        :return: the sample data (in-memory)
        :rtype: ndarray
        """
        with h5py.File(self.path, 'r') as hdf5_file:
            return hdf5_file[self.fov.name][:, :, :, Y, X]
