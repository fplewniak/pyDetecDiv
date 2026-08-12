#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 Class to manipulate Image resources: loading data from files, etc
"""
import abc

import torch
import numpy as np
import pandas as pd
from bioio_base.dimensions import Dimensions

from pydetecdiv.domain.Image import Image, ImgDType


class ImageResourceData(abc.ABC):
    """
    An abstract class to access image resources (files) on disk without having to load whole time series into memory
    """
    image_resource = None
    fov = None
    _drift = None

    @property
    def drift(self) -> pd.DataFrame | None:
        """
        Returns the drift values (dX and dY) for each frame. These values are stored in a csv file that can be loaded into a pandas
        DataFrame.
        """
        if self._drift is None:
            self._drift = self.fov.project.get_object('ImageResource', self.image_resource, use_pool=False).drift
        return self._drift

    @property
    def drift_method(self) -> str:
        """
        Returns the drift method used to correct drift
        """
        return self.fov.image_resource().drift_method

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, int, int, int, int]:
        """
        The image resource shape (should habitually be 5D with the following dimensions TCZYX)
        """

    @property
    @abc.abstractmethod
    def dims(self) -> Dimensions:
        """
        The image resource dimensions with their size
        """

    @property
    @abc.abstractmethod
    def sizeT(self) -> int:
        """
        The number of time frames
        """

    @property
    @abc.abstractmethod
    def sizeC(self) -> int:
        """
        The number of channels
        """

    @property
    @abc.abstractmethod
    def sizeZ(self) -> int:
        """
        The number of layers
        """

    @property
    @abc.abstractmethod
    def sizeY(self):
        """
        The image height
        """

    @property
    @abc.abstractmethod
    def sizeX(self) -> int:
        """
        the image width
        """

    @abc.abstractmethod
    def _image(self, C: int = 0, Z: int = 0, T: int = 0, sliceX: slice | None = None, sliceY: slice | None = None,
               drift: bool = False, imgdtype=ImgDType.uint16) -> np.ndarray:
        """
        A 2D grayscale image (one frame, one channel and one layer)

        :param C: the channel index
        :type C: int
        :param Z: the layer index
        :type Z: int
        :param T: the frame index
        :type T: int
        :return: a 2D data array
        :rtype: 2D numpy.array
        """

    def image(self, sliceX: slice | None = None, sliceY: slice | None = None, C: int = 0, **kwargs) -> np.ndarray:
        """
        The in-memory image
        :param sliceX: X slice
        :param sliceY: Y slice
        :param C: channel
        :param kwargs: arguments passed to private method _image
        :return: the image array
        """
        if C is None:
            if sliceX and sliceY:
                return np.zeros((sliceY.stop - sliceY.start, sliceX.stop - sliceX.start), np.uint16)
            return np.zeros((self.sizeY, self.sizeX), np.uint16)
        if sliceX and sliceY:
            return self._image(C=C, **kwargs)[sliceY, sliceX]
        return self._image(C=C, **kwargs)

    @abc.abstractmethod
    def _image_memmap(self, sliceX: slice | None = None, sliceY: slice | None = None, C: int = 0, Z: int = 0, T: int = 0,
                      drift: bool = False) -> np.ndarray:

        """
        A 2D grayscale memory mapped image (one frame, one channel and one layer)

        :param C: the channel index
        :type C: int
        :param Z: the layer index
        :type Z: int
        :param T: the frame index
        :type T: int
        :return: a 2D data array
        :rtype: 2D numpy.array
        """

    def image_memmap(self, sliceX: slice | None = None, sliceY: slice | None = None, **kwargs) -> np.ndarray:
        """
        Memory mapped image
        :param sliceX: X slice
        :param sliceY: Y slice
        :param kwargs: arguments passed to the private method _image_memmap
        :return: the image as a ndarray
        """
        if sliceX and sliceY:
            return self._image_memmap(sliceX=sliceX, sliceY=sliceY, **kwargs)
        return self._image_memmap(**kwargs)

    def refresh(self) -> None:
        """
        A method to refresh memory mapped files (close and reopen) if max memory is used or do nothing for others
        """

    def auto_channels(self, C: int | list[int] | tuple[int] = 0, T: int = 0, Z: int | list[int] | tuple[int] = 0,
                      crop: tuple[slice, slice] | None = None, drift: bool = False, alpha: bool = False,
                      resize: tuple[int, int] | None = None) -> Image:
        """
        Returns a RGB, RGBA or grayscale image depending upon the C or Z values. If C (or Z) is a tuple, it is used as
        RGB values. If alpha is set to True, then the maximum value of every pixel across all channels defines its
        alpha value. If C and Z are both an index, then the returned image is grayscale.

        :param C: the channel or channels tuple
        :param T: the time frame index
        :param Z: the z-slice or z-slices tuple
        :param crop: a tuple defining the crop values as slices = (slice(xmin, xmax), slice(ymin, ymax))
        :param drift: bool defining whether drift correction should be applied
        :param alpha: bool defining whether the image should contain an alpha channel
        :param resize: the new size of the image if resizing is requested
        """
        if crop is None:
            crop = (None, None)
        if isinstance(C, int):
            if isinstance(Z, (tuple, list)):
                return Image.compose_channels(
                        [Image(self.image(C=C, T=T, Z=c, sliceX=crop[0], sliceY=crop[1], drift=drift)).resize(shape=resize) for c
                         in Z], alpha=alpha)
            else:
                return Image(self.image(C=C, T=T, Z=Z, sliceX=crop[0], sliceY=crop[1], drift=drift)).resize(shape=resize)
        elif isinstance(C, (tuple, list)):
            if isinstance(Z, (tuple, list)):
                return Image.compose_channels(
                        [Image(self.image(C=c, T=T, Z=Z[0], sliceX=crop[0], sliceY=crop[1], drift=drift)).resize(shape=resize) for c in
                         C], alpha=alpha)
        return Image.compose_channels(
                [Image(self.image(C=c, T=T, Z=Z, sliceX=crop[0], sliceY=crop[1], drift=drift)).resize(shape=resize) for c in C],
                alpha=alpha)

    def sequence(self, seqlen: int,
                 C: int = 0, T: int = 0, Z: int | list[int] | tuple[int] = 0, resize: tuple[int, int] | None = None,
                 crop: tuple[slice, slice] | None = None, drift: bool = False, alpha: bool = False) -> torch.Tensor:
        img = self.auto_channels(C=C, T=T, Z=Z, crop=crop, drift=drift, alpha=alpha, resize=resize)
        sequence = img.as_tensor().unsqueeze(dim=0)
        for frame in range(T + 1, T + seqlen):
            img = self.auto_channels(C=C, T=frame, Z=Z, crop=crop, drift=drift, alpha=alpha, resize=resize)
            sequence = torch.cat([sequence, img.as_tensor().unsqueeze(dim=0)], dim=0)
        return sequence
