"""
 A class handling image resource in multiple files (one for each combination of T, C, Z dimensions)
"""
from typing import TYPE_CHECKING

from ndtiff import NDTiffDataset
import numpy as np
from bioio_base.dimensions import Dimensions
import cv2

from pydetecdiv.domain.ImageResourceData import ImageResourceData
from pydetecdiv.domain.Image import Image, ImgDType

if TYPE_CHECKING:
    from pydetecdiv.domain.ImageResource import ImageResource


class NDTiffImageResource(ImageResourceData):
    """
    A business-logic class defining valid operations and attributes of Image resources stored in multiple files
    """

    def __init__(self, image_resource: 'ImageResource' = None):
        # self.image_files = image_resource.image_files_5d
        self.path = image_resource.image_files
        self.fov = image_resource.fov
        self.pos_index = image_resource.key_val['pos_index']
        self.image_resource = image_resource.id_
        self._shape = image_resource.shape
        self._dims = image_resource.dims
        self._drift = image_resource.drift
        self._ndtiff_ds = None
        self._dask_array = None

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

    @property
    def ndtiff_ds(self) -> NDTiffDataset:
        """
        Property returning the NDTiff dataset associated with this image resource, creating it if it has not been yet
        :return: the NDTiff dataset
        """
        if self._ndtiff_ds is None:
            self._ndtiff_ds = NDTiffDataset(self.path[0])
        return self._ndtiff_ds

    def channel_list(self, channel: int | list | tuple | None = None, z: int | None = None, time: int | None = None,
                     sliceX: slice | None = None, sliceY: slice | None = None, drift: bool = False,
                     alpha: bool = False, resize: tuple[int, int] | None = None) -> list:
        """
        Return data for a list of channels
        :param channel: the channel or list of channels
        :param z: the z stack
        :param time: the time frame
        :param sliceX: the X slice
        :param sliceY: the Y slice
        :param drift: True if drift correction must be applied
        :param alpha: True if alpha layer should be included
        :param resize: the new size
        """
        if not isinstance(channel, (tuple, list)):
            channel = [channel]
        img_list = []
        data = None
        for c in channel:
            if c is not None:
                data = self.dask_array[time, c, z][sliceY, sliceX].compute().squeeze()
            else:
                if sliceX.start and sliceY.start:
                    data = np.zeros((sliceY.stop - sliceY.start, sliceX.stop - sliceX.start), np.uint16)
                else:
                    data = np.zeros((self.sizeY, self.sizeX), np.uint16)
            if drift and self.drift is not None:
                data = cv2.warpAffine(np.array(data),
                                      np.float32(
                                              [[1, 0, -self.drift.iloc[time].dx],
                                               [0, 1, -self.drift.iloc[time].dy]]),
                                      (data.shape[1], data.shape[0]))
            img_list.append(Image(data).resize(shape=resize))
        return img_list

    @property
    def dask_array(self):
        """
        The image as a dask array
        :return:
        """
        if self._dask_array is None:
            self._dask_array = self.ndtiff_ds.as_array(['position', 'time', 'channel', 'z'])[self.pos_in_array]
        return self._dask_array

    @property
    def pos_in_array(self) -> int:
        """
        The position in the dask array
        """
        return self.ndtiff_ds.axes['position'].index(self.pos_index)

    def _image(self, C: int = 0, Z: int = 0, T: int = 0, sliceX: slice = None, sliceY: slice = None,
               drift: bool = False, imgdtype=ImgDType.uint16) -> np.ndarray:
        """
        A 2D grayscale image (on frame, one channel and one layer)

        :param C: the channel index
        :param Z: the layer index
        :param T: the frame index
        :param drift: True if the drift correction should be applied
        :return: a 2D data array
        """
        if self.ndtiff_ds.has_image(channel=C, z=Z, time=T, position=self.pos_index):
            data = self.ndtiff_ds.read_image(channel=C, z=Z, time=T, position=self.pos_index)
            if drift and self.drift is not None:
                data = cv2.warpAffine(np.array(data),
                                      np.float32(
                                              [[1, 0, -self.drift.iloc[T].dx],
                                               [0, 1, -self.drift.iloc[T].dy]]),
                                      (data.shape[1], data.shape[0]))
            # data = tf.image.convert_image_dtype(data, dtype=tf.uint16, saturate=False).numpy()
            data = Image(data).as_array(dtype=imgdtype)
            if sliceX and sliceY:
                return data[sliceY, sliceX]
            return data
        if sliceX and sliceY:
            return np.zeros((sliceY.stop - sliceY.start, sliceX.stop - sliceX.start), imgdtype)
        return np.zeros((self.sizeY, self.sizeX), imgdtype)

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

        return self._image(C, Z, T)[sliceY, sliceX]

    def auto_channels(self, C: int = 0, T: int = 0, Z: int | list[int] | tuple[int] = 0,
                      crop: tuple[slice, slice] = None, drift: bool = False, alpha: bool = False, resize: tuple[int, int] = None) -> Image:
        """
        Returns a RGB, RGBA or grayscale image depending upon the C or Z values. If C (or Z) is a tuple, it is used as
        RGB values. If alpha is set to True, then the maximum value of every pixel across all channels defines its
        alpha value. If C and Z are both an index, then the returned image is grayscale.

        :param image_resource_data: the image resource data used to create the Image
        :param C: the channel or channels tuple
        :param T: the time frame index
        :param Z: the z-slice or z-slices tuple
        :param crop: a tuple defining the crop values as slices = (slice(xmin, xmax), slice(ymin, ymax))
        :param drift: bool defining whether drift correction should be applied
        :param alpha: bool defining whether the image should contain an alpha channel
        :return: Image
        """
        img = None

        if crop is not None:
            sliceX = crop[0]
            sliceY = crop[1]
        else:
            sliceX = slice(None, None)
            sliceY = slice(None, None)

        if isinstance(C, int):
            # data = self.dask_array[T][C][Z][sliceY, sliceX].compute()
            data = self.dask_array[T, C, Z, sliceY, sliceX].compute()
            if drift and self.drift is not None:
                data = cv2.warpAffine(np.array(data),
                                      np.float32(
                                              [[1, 0, -self.drift.iloc[T].dx],
                                               [0, 1, -self.drift.iloc[T].dy]]),
                                      (data.shape[1], data.shape[0]))
            img = Image(data).resize(shape=resize)
        elif isinstance(C, (tuple, list)):
            img = Image.compose_channels(self.channel_list(channel=C, z=Z, time=T, sliceX=sliceX, sliceY=sliceY, drift=drift),
                                         alpha=alpha)
        return img
