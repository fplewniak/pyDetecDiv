"""
 A class handling image resource in multiple files (one for each combination of T, C, Z dimensions)
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydetecdiv.domain.ImageResource import ImageResource

from bioio_base.dimensions import Dimensions

import numpy as np
from tifffile import tifffile
import cv2
from pydetecdiv.domain.ImageResourceData import ImageResourceData
from pydetecdiv.domain.Image import Image, ImgDType


# def aics_indexer(path: str, pattern: str) -> pd.Series:
#     """
#     An indexer to determine dimensions (T, C, Z) from file names to be used with AICSImage reader when reading a list of
#     files
#
#     :param path: the path of the file name
#     :type path: str
#     :param pattern: the pattern defining the dimension indexes from file names
#     :type pattern: regex str
#     :return: the dimension indexes corresponding to the path
#     :rtype: pandas Series
#     """
#     return pd.Series({k: int(v) for k, v in re.search(pattern, path).groupdict().items()})


class MultiFileImageResource(ImageResourceData):
    """
    A business-logic class defining valid operations and attributes of Image resources stored in multiple files
    """

    def __init__(self, max_mem: int = 5000, image_resource: 'ImageResource' = None):
        self.image_files = image_resource.image_files_5d
        self.path = image_resource.image_files
        self.pattern = image_resource.pattern
        self.fov = image_resource.fov
        self.image_resource = image_resource.id_
        self.max_mem = max_mem
        self._shape = image_resource.shape
        self._dims = image_resource.dims
        self._drift = image_resource.drift

        # print(f'Multiple file image resource: {self.dims}')

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """
        The image resource shape (should be 5D with the following dimensions TCZYX)
        """
        return self._shape

    @property
    def dims(self) -> Dimensions:
        """
        the dimensions of the image resource
        :return:
        """
        return self._dims

    @property
    def sizeT(self) -> int:
        """
        The number of frames
        """
        return self._dims.T

    @property
    def sizeC(self) -> int:
        """
        The number of channels
        """
        return self._dims.C

    @property
    def sizeZ(self) -> int:
        """
        The number of layers
        """
        return self._dims.Z

    @property
    def sizeY(self) -> int:
        """
        The height of image
        """
        return self._dims.Y

    @property
    def sizeX(self) -> int:
        """
        The width of image
        """
        return self._dims.X

    def _image(self, C: int = 0, Z: int = 0, T: int = 0, drift: bool = False) -> np.ndarray:
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
        if self.image_files[T, C, Z]:
            data = tifffile.imread(self.image_files[T, C, Z])
            if drift and self.drift is not None:
                data = cv2.warpAffine(np.array(data),
                                      np.float32(
                                              [[1, 0, -self.drift.iloc[T].dx],
                                               [0, 1, -self.drift.iloc[T].dy]]),
                                      (data.shape[1], data.shape[0]))
            # data = tf.image.convert_image_dtype(data, dtype=tf.uint16, saturate=False).numpy()
            data = Image(data).as_array(dtype=ImgDType.uint16)
            return data
        return np.zeros((self.sizeY, self.sizeX), np.uint16)

    def _image_memmap(self, sliceX: slice = None, sliceY: slice = None, C: int = 0, Z: int = 0, T: int = 0,
                      drift: bool = False) -> np.ndarray:
        if sliceX is None:
            sliceX = slice(0, self.sizeX)
        if sliceY is None:
            sliceY = slice(0, self.sizeX)
        deltaX = 0 if not drift or self.drift is None else int(round(self.drift.iloc[T].dx))
        deltaY = 0 if not drift or self.drift is None else int(round(self.drift.iloc[T].dy))

        sliceX = slice(sliceX.start + deltaX, sliceX.stop + deltaX)
        sliceY = slice(sliceY.start + deltaY, sliceY.stop + deltaY)

        if self.image_files[T, C, Z]:
            return tifffile.memmap(self.image_files[T, C, Z])[sliceY, sliceX]
        return np.zeros((sliceY.stop - sliceY.start, sliceX.stop - sliceX.start), np.uint16)

    # def data_sample(self, X: slice = None, Y: slice = None) -> np.ndarray:
    #     """
    #     Return a sample from an image resource, specified by X and Y slices. This is useful to extract resources for
    #     regions of interest from a field of view.
    #
    #     :param X: the X slice
    #     :type X: slice
    #     :param Y: the Y slice
    #     :type Y: slice
    #     :return: the sample data (in-memory)
    #     :rtype: ndarray
    #     """
    #     return (AICSImage(self.path, indexer=lambda x: aics_indexer(x, self.pattern)).reader
    #             .get_image_dask_data('TCZYX', X=X, Y=Y).compute())
