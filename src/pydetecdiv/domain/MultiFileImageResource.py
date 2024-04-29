"""
 A class handling image resource in multiple files (one for each combination of T, C, Z dimensions)
"""
import re
import tensorflow as tf
from aicsimageio import AICSImage
import numpy as np
import pandas as pd
from tifffile import tifffile
import cv2
from pydetecdiv.domain import ImageResourceData


def aics_indexer(path, pattern):
    """
    An indexer to determine dimensions (T, C, Z) from file names to be used with AICSImage reader when reading a list of
    files

    :param path: the path of the file name
    :type path: str
    :param pattern: the pattern defining the dimension indexes from file names
    :type pattern: regex str
    :return: the dimension indexes corresponding to the path
    :rtype: pandas Series
    """
    return pd.Series({k: int(v) for k, v in re.search(pattern, path).groupdict().items()})


class MultiFileImageResource(ImageResourceData):
    """
    A business-logic class defining valid operations and attributes of Image resources stored in multiple files
    """

    def __init__(self, max_mem=5000, image_resource=None):
        self.image_files = image_resource.image_files_5d
        self.path = image_resource.image_files
        self.pattern = image_resource.pattern
        self.fov = image_resource.fov
        self.image_resource = image_resource.id_
        self.max_mem = max_mem
        self._shape = image_resource.shape
        self._dims = image_resource.dims

        # print(f'Multiple file image resource: {self.dims}')

    @property
    def shape(self):
        """
        The image resource shape (should be 5D with the following dimensions TCZYX)
        """
        return self._shape

    @property
    def dims(self):
        """
        the dimensions of the image resource
        :return:
        """
        return self._dims

    @property
    def sizeT(self):
        """
        The number of frames
        """
        return self._dims.T

    @property
    def sizeC(self):
        """
        The number of channels
        """
        return self._dims.C

    @property
    def sizeZ(self):
        """
        The number of layers
        """
        return self._dims.Z

    @property
    def sizeY(self):
        """
        The height of image
        """
        return self._dims.Y

    @property
    def sizeX(self):
        """
        The width of image
        """
        return self._dims.X

    def _image(self, C=0, Z=0, T=0, drift=None):
        """
        A 2D grayscale image (on frame, one channel and one layer)

        :param C: the channel index
        :type C: int
        :param Z: the layer index
        :type Z: int
        :param T: the frame index
        :type T: int
        :return: a 2D data array
        :rtype: 2D numpy.array
        """
        if self.image_files[T, C, Z]:
            data = tifffile.imread(self.image_files[T, C, Z])
            if drift is not None:
                data = cv2.warpAffine(np.array(data),
                                      np.float32(
                                          [[1, 0, -drift.dx],
                                           [0, 1, -drift.dy]]),
                                      (data.shape[1], data.shape[0]))
            data = tf.image.convert_image_dtype(data, dtype=tf.uint16, saturate=False).numpy()
            return data
        return np.zeros((self.sizeY, self.sizeX), np.uint16)
        # return None

    def _image_memmap(self,  sliceX=None, sliceY=None, C=0, Z=0, T=0, drift=None):
        if sliceX is None:
            sliceX = slice(0, self.sizeX)
        if sliceY is None:
            sliceY = slice(0, self.sizeX)
        deltaX = 0 if drift is None else drift.dx
        deltaY = 0 if drift is None else drift.dy

        sliceX = slice(sliceX.start - deltaX, sliceX.stop - deltaX)
        sliceY = slice(sliceY.start - deltaY, sliceY.stop - deltaY)

        if self.image_files[T, C, Z]:
            return tifffile.memmap(self.image_files[T, C, Z])[sliceY, sliceX]
        return np.zeros((sliceY.stop - sliceY.start, sliceX.stop - sliceX.start), np.uint16)


    def data_sample(self, X=None, Y=None):
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
        return (AICSImage(self.path, indexer=lambda x: aics_indexer(x, self.pattern)).reader
                .get_image_dask_data('TCZYX', X=X, Y=Y).compute())
