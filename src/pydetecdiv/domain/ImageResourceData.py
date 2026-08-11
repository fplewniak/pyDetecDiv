#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 Class to manipulate Image resources: loading data from files, etc
"""
import os
import abc

import torch
from PySide6.QtCore import QThread
import numpy as np
import pandas as pd
import cv2 as cv
from bioio_base.dimensions import Dimensions
from vidstab import VidStab

from pydetecdiv.domain.Image import Image
from pydetecdiv.settings import get_config_value


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
    def shape(self) -> tuple[int]:
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
    def _image(self, C: int = 0, Z: int = 0, T: int = 0, sliceX: slice = None, sliceY: slice = None,
               drift: bool = False) -> np.ndarray:
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

    def image(self, sliceX: slice = None, sliceY: slice = None, C: int = 0, **kwargs) -> np.ndarray:
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
                # return np.zeros((self.sizeY, self.sizeX), np.uint16)[sliceY, sliceX]
                return np.zeros((sliceY.stop - sliceY.start, sliceX.stop - sliceX.start), np.uint16)
            return np.zeros((self.sizeY, self.sizeX), np.uint16)
        if sliceX and sliceY:
            return self._image(C=C, **kwargs)[sliceY, sliceX]
        return self._image(C=C, **kwargs)

    @abc.abstractmethod
    def _image_memmap(self, sliceX: slice = None, sliceY: slice = None, C: int = 0, Z: int = 0, T: int = 0,
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

    def image_memmap(self, sliceX: slice = None, sliceY: slice = None, **kwargs) -> np.ndarray:
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

    # @abc.abstractmethod
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

    # def open(self):
    #     """
    #     Open the memory mapped file to access data
    #     """
    #
    # def close(self):
    #     """
    #     Close the memory mapped file
    #     """
    #
    # def flush(self):
    #     """
    #     Flush the data to save changes to the meory mapped file
    #     """
    #
    # def refresh(self):
    #     """
    #     Close and open the memory mapped file to save memory if needed. Useful when creating a new file or making lots
    #     of changes.
    #     """

    # def compute_drift(self, method: str = 'phase correlation', **kwargs) -> pd.DataFrame:
    #     """
    #     Compute drift along time using the specified method
    #
    #     :param method: the method to compute drift
    #     :type method: str
    #     :param kwargs: keyword arguments passed to the drift computation method
    #     :return: the cumulative transforms for drift correction
    #     :rtype: pandas DataFrame
    #     """
    #     match (method):
    #         case 'phase correlation':
    #             drift = pd.concat([pd.DataFrame([[0, 0]], columns=['dx', 'dy']),
    #                                self.compute_drift_phase_correlation_cv2(**kwargs)], ignore_index=True)
    #         case 'vidstab':
    #             drift = pd.concat([pd.DataFrame([[0, 0]], columns=['dx', 'dy']),
    #                                self.compute_drift_vidstab(**kwargs)], ignore_index=True)
    #         case _:
    #             drift = pd.DataFrame([[0, 0]] * self.sizeT, columns=['dx', 'dy'])
    #     image_resource = self.fov.image_resource()
    #     if image_resource.key_val is None:
    #         image_resource.key_val = {}
    #     drift_file = f'{self.fov.name}_drift_data.csv'
    #     drift_path = os.path.join(get_config_value('project', 'workspace'),
    #                               self.fov.project.dbname, drift_file)
    #     drift.to_csv(drift_path, float_format='%.3f', columns=['dx', 'dy'], index=False)
    #     self.fov.set_timestamp()
    #     image_resource.key_val.update({'drift': drift_file, 'drift method': method})
    #     image_resource.validate()
    #     image_resource.project.commit()
    #     return drift
    #
    # def compute_drift_phase_correlation_cv2(self, Z: int = 0, C: int = 0, thread: QThread = None) -> pd.DataFrame | None:
    #     """
    #     Compute the cumulative transforms (dx, dy) to apply in order to correct the drift using phase correlation
    #
    #     :param Z: the layer index
    #     :type Z: int
    #     :param C: the channel index
    #     :type C: int
    #     :param max_mem: maximum memory use when using memory mapped TIFF
    #     :type max_mem: int
    #     :return: the cumulative drift transforms dx, dy, dr
    #     :rtype: pandas DataFrame
    #     """
    #     df = pd.DataFrame(columns=['dx', 'dy'])
    #     for frame in range(1, self.sizeT):
    #         df.loc[len(df)], _ = cv.phaseCorrelate(np.float32(self.image(T=frame - 1, Z=Z, C=C)),
    #                                                np.float32(self.image(T=frame, Z=Z, C=C)))
    #         self.refresh()
    #         if thread and thread.isInterruptionRequested():
    #             return None
    #     return df.cumsum(axis=0)
    #
    # def compute_drift_vidstab(self, Z: int = 0, C: int = 0, thread: QThread = None,
    #                           smoothing_window: int = 1) -> pd.DataFrame | None:
    #     """
    #     Compute the cumulative transforms (dx, dy, dr) to apply in order to stabilize the time series and correct drift
    #
    #     :param Z: the layer index
    #     :type Z: int
    #     :param C: the channel index
    #     :type C: int
    #     :param max_mem: maximum memory use when using memory mapped TIFF
    #     :type max_mem: int
    #     :return: the cumulative drift transforms dx, dy, dr
    #     :rtype: pandas DataFrame
    #     """
    #     stabilizer = VidStab()
    #     for frame in range(0, self.sizeT):
    #         _ = stabilizer.stabilize_frame(
    #                 input_frame=np.uint8(np.array(self.image(T=frame, Z=Z, C=C)) / 65535 * 255), smoothing_window=smoothing_window)
    #         self.refresh()
    #         if thread and thread.isInterruptionRequested():
    #             return None
    #     return pd.DataFrame(stabilizer.transforms, columns=('dx', 'dy', 'dr')).cumsum(axis=0)[['dx', 'dy']]

    def refresh(self) -> None:
        """
        A method to refresh memory mapped files (close and reopen) if max memory is used or do nothing for others
        """

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
        if crop is None:
            crop = (None, None)
        if isinstance(C, int):
            if isinstance(Z, (tuple, list)):
                img = Image.compose_channels(
                        [Image(self.image(C=C, T=T, Z=c, sliceX=crop[0], sliceY=crop[1], drift=drift)).resize(shape=resize) for c
                         in Z], alpha=alpha)
            else:
                img = Image(self.image(C=C, T=T, Z=Z, sliceX=crop[0], sliceY=crop[1], drift=drift)).resize(shape=resize)
        elif isinstance(C, (tuple, list)):
            img = Image.compose_channels(
                    [Image(self.image(C=c, T=T, Z=Z, sliceX=crop[0], sliceY=crop[1], drift=drift)).resize(shape=resize) for c in
                     C], alpha=alpha)
        return img

    def sequence(self, seqlen: int,
                 C: int = 0, T: int = 0, Z: int | list[int] | tuple[int] = 0, resize: tuple[int, int] = None,
                 crop: tuple[slice, slice] = None, drift: bool = False, alpha: bool = False) -> torch.Tensor:
        sequence = None
        for frame in range(T, T + seqlen):
            img = self.auto_channels(C=C, T=frame, Z=Z, crop=crop, drift=drift, alpha=alpha, resize=resize)
            if sequence is None:
                sequence = img.as_tensor().unsqueeze(dim=0)
            else:
                sequence = torch.cat([sequence, img.as_tensor().unsqueeze(dim=0)], dim=0)
        return sequence
