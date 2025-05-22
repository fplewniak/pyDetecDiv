"""
 A class handling image resource in a data array
"""
from typing import TYPE_CHECKING

from bioio_base.dimensions import Dimensions

if TYPE_CHECKING:
    from pydetecdiv.domain import ImageResource

from bioio import BioImage
import numpy as np
import cv2

from pydetecdiv.domain import ImageResourceData


class ArrayImageResource(ImageResourceData):
    """
    A business-logic class defining valid operations and attributes of Image resources stored in a data array
    """
    def __init__(self, data: np.ndarray = None, image_resource: 'ImageResource' = None, max_mem=5000):
        self.img_data = BioImage(data)
        self.fov = image_resource.fov
        self.image_resource = image_resource.id_
        self.max_mem = max_mem

        print(f'Array image resource: {self.dims}')

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The image resource shape (should be 5D with the following dimensions TCZYX)
        """
        return self.img_data.shape

    @property
    def dims(self) -> Dimensions:
        """
        the dimensions of the image resource
        """
        return self.img_data.dims

    @property
    def sizeT(self) -> int:
        """
        The number of frames
        """
        return self.img_data.dims.T

    @property
    def sizeC(self) -> int:
        """
        The number of channels
        """
        return self.img_data.dims.C

    @property
    def sizeZ(self) -> int:
        """
        The number of layers
        """
        return self.img_data.dims.Z

    @property
    def sizeY(self) -> int:
        """
        The height of image
        """
        return self.img_data.dims.Y

    @property
    def sizeX(self) -> int:
        """
        The width of image
        """
        return self.img_data.dims.X

    def image(self, C: int = 0, Z: int = 0, T: int = 0, drift: bool = None) -> np.ndarray:
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
        data = self.img_data.get_image_dask_data('TCZ', X=X, Y=Y).compute()
        return data
