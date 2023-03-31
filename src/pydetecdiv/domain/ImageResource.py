#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 Classes to manipulate Image resources: loading data from files, etc
"""
import glob
import re
import h5py
import numpy as np
import pandas as pd
import skimage.io as skio
import xmltodict
import tifffile
from aicsimageio import AICSImage
# from memory_profiler import profile
import psutil
from tifffile import TiffFile, TiffSequence
import cv2 as cv
from vidstab import VidStab


def aics_indexer(path, pattern):
    return pd.Series({k: int(v) for k, v in re.search(pattern, path).groupdict().items()})


class ImageResource:
    def __init__(self, path, pattern=None, max_mem=5000, **kwargs):
        self.path = path
        self.max_mem = max_mem
        if pattern is None:
            self._memmap = tifffile.memmap(path, **kwargs)
            self.resource = AICSImage(self.path).reader
        else:
            self._memmap = None
            self.path = path if isinstance(path, list) else [path]
            self.resource = AICSImage(self.path, indexer=lambda x: aics_indexer(x, pattern)).reader

    @property
    def shape(self):
        return self.resource.shape

    @property
    def dims(self):
        return self.resource.dims

    @property
    def sizeT(self):
        return self.resource.dims.T

    @property
    def sizeC(self):
        return self.resource.dims.C

    @property
    def sizeZ(self):
        return self.resource.dims.Z

    @property
    def sizeY(self):
        return self.resource.dims.Y

    @property
    def sizeX(self):
        return self.resource.dims.X

    def image(self, C=0, Z=0, T=0):
        if self._memmap is not None:
            s = self.resource.shape
            data = np.expand_dims(self._memmap, axis=tuple([i for i in range(len(s)) if s[i] == 1]))[T, C, Z, ...]
        else:
            data = self.resource.get_image_dask_data('YX', C=C, Z=Z, T=T).compute()
        return data

    def open(self):
        """
        Open the memory mapped file to access data
        """
        self._memmap = tifffile.memmap(self.path)

    def close(self):
        """
        Close the memory mapped file
        """
        if self._memmap is not None:
            if not self._memmap._mmap.closed:
                self._memmap._mmap.close()

    def flush(self):
        """
        Flush the data to save changes to the meory mapped file
        """
        if self._memmap is not None:
            if not self._memmap._mmap.closed:
                self._memmap._mmap.flush()

    def refresh(self):
        """
        Close and open the memory mapped file to save memory if needed. Useful when creating a new file or making lots
        of changes.
        """
        if self._memmap is not None:
            self.close()
            self.open()

    def compute_drift(self, Z=0, C=0, max_mem=5000):
        # if self._memmap is not None:
        #     return self._memmap.compute_drift(z=z, c=c, max_mem=max_mem)
        stabilizer = VidStab()
        for frame in range(0, self.sizeT):
            _ = stabilizer.stabilize_frame(
                input_frame=np.uint8(np.array(self.image(T=frame, Z=Z, C=C)) / 65535 * 255), smoothing_window=1)
            if psutil.Process().memory_info().rss / (1024 * 1024) > max_mem:
                self.refresh()
        return pd.DataFrame(stabilizer.transforms, columns=('dx', 'dy', 'dr')).cumsum(axis=0)

    def correct_drift(self, drift, filename=None, max_mem=5000):
        new_image = ImageResource(filename, max_mem=max_mem, mode='readwrite', ome=True,
                                  metadata={'axes': self.dims.order}, shape=self.shape, dtype=np.uint16)
        new_memmap = new_image._memmap
        for c in range(0, self.sizeC):
            for z in range(0, self.sizeZ):
                new_memmap[0, c, z, ...] = self.image(C=c, T=0, Z=z)
                for idx in drift.index:
                    new_memmap[idx + 1, c, z, ...] = cv.warpAffine(np.array(self.image(C=c, T=idx + 1, Z=z)),
                                                                   np.float32(
                                                                       [[1, 0, -drift.iloc[idx].dx],
                                                                        [0, 1, -drift.iloc[idx].dy]]),
                                                                   self.image(C=c, T=idx + 1, Z=z).shape)

                    if psutil.Process().memory_info().rss / (1024 * 1024) > max_mem:
                        self.refresh()
                        new_image.refresh()
