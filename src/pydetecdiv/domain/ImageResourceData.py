#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 Classes to manipulate Image resources: loading data from files, etc
"""
import numpy as np
import pandas as pd
import cv2 as cv
from vidstab import VidStab


# def aics_indexer(path, pattern):
#     """
#     An indexer to determine dimensions (T, C, Z) from file names to be used with AICSImage reader when reading a
#     list of files
#
#     :param path: the path of the file name
#     :type path: str
#     :param pattern: the pattern defining the dimension indexes from file names
#     :type pattern: regex str
#     :return: the dimension indexes corresponding to the path
#     :rtype: pandas Series
#     """
#     return pd.Series({k: int(v) for k, v in re.search(pattern, path).groupdict().items()})


class ImageResourceData:
    """
    A generic class to access image resources (files) on disk without having to load whole time series into memory
    """
    image_resource = None

    @property
    def shape(self):
        """
        The image resource shape (should habitually be 5D with the following dimensions TCZYX)
        """
        return (-1, -1, -1, -1, -1)

    @property
    def dims(self):
        """
        The image resource dimensions with their size
        """
        return -1

    @property
    def sizeT(self):
        """
        The number of time frames
        """
        return -1

    @property
    def sizeC(self):
        """
        The number of channels
        """
        return -1

    @property
    def sizeZ(self):
        """
        The number of layers
        """
        return -1

    @property
    def sizeY(self):
        """
        The image height
        """
        return -1

    @property
    def sizeX(self):
        """
        the image width
        """
        return -1

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
        return np.empty((self.image_resource.sizeY, self.image_resource.sizeX))

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
        return np.empty((Y[1] - Y[0], X[1] - X[0]))

    def open(self):
        """
        Open the memory mapped file to access data
        """


    def close(self):
        """
        Close the memory mapped file
        """


    def flush(self):
        """
        Flush the data to save changes to the meory mapped file
        """


    def refresh(self):
        """
        Close and open the memory mapped file to save memory if needed. Useful when creating a new file or making lots
        of changes.
        """


    def compute_drift(self, method='phase correlation', **kwargs):
        """
        Compute drift along time using the specified method

        :param method: the method to compute drift
        :type method: str
        :param kwargs: keyword arguments passed to the drift computation method
        :return: the cumulative transforms for drift correction
        :rtype: pandas DataFrame
        """
        match (method):
            case 'phase correlation':
                return self.compute_drift_phase_correlation_cv2(**kwargs)
            case 'vidstab':
                return self.compute_drift_vidstab(**kwargs)
            case _:
                return pd.DataFrame([[0, 0]] * self.sizeT, columns=['dy', 'dx'])

    def compute_drift_phase_correlation_cv2(self, Z=0, C=0, thread=None):
        """
        Compute the cumulative transforms (dx, dy) to apply in order to correct the drift using phase correlation

        :param Z: the layer index
        :type Z: int
        :param C: the channel index
        :type C: int
        :param max_mem: maximum memory use when using memory mapped TIFF
        :type max_mem: int
        :return: the cumulative drift transforms dx, dy, dr
        :rtype: pandas DataFrame
        """
        df = pd.DataFrame(columns=['dx', 'dy'])
        for frame in range(1, self.sizeT):
            df.loc[len(df)], _ = cv.phaseCorrelate(np.float32(self.image(T=frame - 1, Z=Z, C=C)),
                                                   np.float32(self.image(T=frame, Z=Z, C=C)))
            self.refresh()
            if thread and thread.isInterruptionRequested():
                return None
        return df.cumsum(axis=0)

    def compute_drift_vidstab(self, Z=0, C=0, thread=None):
        """
        Compute the cumulative transforms (dx, dy, dr) to apply in order to stabilize the time series and correct drift

        :param Z: the layer index
        :type Z: int
        :param C: the channel index
        :type C: int
        :param max_mem: maximum memory use when using memory mapped TIFF
        :type max_mem: int
        :return: the cumulative drift transforms dx, dy, dr
        :rtype: pandas DataFrame
        """
        stabilizer = VidStab()
        for frame in range(0, self.sizeT):
            _ = stabilizer.stabilize_frame(
                input_frame=np.uint8(np.array(self.image(T=frame, Z=Z, C=C)) / 65535 * 255), smoothing_window=1)
            self.refresh()
            if thread and thread.isInterruptionRequested():
                return None
        return pd.DataFrame(stabilizer.transforms, columns=('dx', 'dy', 'dr')).cumsum(axis=0)

    # def correct_drift(self, drift, filename=None, max_mem=5000):
    #     """
    #     Apply the drift correction and save to a multipage TIF file
    #
    #     :param drift: the cumulative transforms to apply
    #     :type drift: pandas DataFrame
    #     :param filename: the file name to save the stabilized time series to
    #     :type filename: str
    #     :param max_mem: maximum memory allowed use
    #     :type max_mem: int
    #     """
    #     new_image = ImageResourceData(filename, max_mem=max_mem, mode='readwrite', ome=True,
    #                                   metadata={'axes': self.dims.order}, shape=self.shape, dtype=np.uint16)
    #     new_memmap = new_image._memmap
    #     for c in range(0, self.sizeC):
    #         for z in range(0, self.sizeZ):
    #             new_memmap[0, c, z, ...] = self.image(C=c, T=0, Z=z)
    #             for idx in drift.index:
    #                 new_memmap[idx + 1, c, z, ...] = cv.warpAffine(np.array(self.image(C=c, T=idx + 1, Z=z)),
    #                                                                np.float32(
    #                                                                    [[1, 0, -drift.iloc[idx].dx],
    #                                                                     [0, 1, -drift.iloc[idx].dy]]),
    #                                                                self.image(C=c, T=idx + 1, Z=z).shape)
    #
    #                 if psutil.Process().memory_info().rss / (1024 * 1024) > max_mem:
    #                     self.refresh()
    #                     new_image.refresh()
