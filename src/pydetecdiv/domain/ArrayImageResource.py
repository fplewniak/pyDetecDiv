"""
 A class handling image resource in a data array
"""
from aicsimageio import AICSImage
import numpy as np
import cv2

from pydetecdiv.domain.ImageResourceData import ImageResourceData


class ArrayImageResource(ImageResourceData):
    """
    A business-logic class defining valid operations and attributes of Image resources stored in a data array
    """
    def __init__(self, data=None, image_resource=None, max_mem=5000):
        self.img_data = AICSImage(data)
        self.fov = image_resource.fov
        self.image_resource = image_resource.id_
        self.max_mem = max_mem

        print(f'Array image resource: {self.dims}')

    @property
    def shape(self):
        """
        The image resource shape (should be 5D with the following dimensions TCZYX)
        """
        return self.img_data.shape

    @property
    def dims(self):
        return self.img_data.dims

    @property
    def sizeT(self):
        """
        The number of channels
        """
        return self.img_data.dims.T

    @property
    def sizeC(self):
        """
        The number of channels
        """
        return self.img_data.dims.C

    @property
    def sizeZ(self):
        """
        The number of channels
        """
        return self.img_data.dims.Z

    @property
    def sizeY(self):
        """
        The number of channels
        """
        return self.img_data.dims.Y

    @property
    def sizeX(self):
        """
        The number of channels
        """
        return self.img_data.dims.X

    def image(self, C=0, Z=0, T=0, drift=None):
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
        data = self.img_data.get_image_dask_data('YX', C=C, Z=Z, T=T).compute()

        if drift is not None:
            return cv2.warpAffine(np.array(data),
                                  np.float32(
                                      [[1, 0, -drift.dx],
                                       [0, 1, -drift.dy]]),
                                  (data.shape[1], data.shape[0]))
        return data

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
        data = self.img_data.get_image_dask_data('TCZ', X=X, Y=Y).compute()
        return data
