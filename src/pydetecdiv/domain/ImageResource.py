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

class ImageResource:
    def __init__(self, path, mode='readwrite', **kwargs):
        self.path = path
        self._aics_image = AICSImage(path, **kwargs)
        try:
            self._memmap = MemMapTiff(path, mode=mode)
        except ValueError:
            self._memmap = None

    def image(self, c=0, z=0, t=0, **kwargs):
        xd = self._aics_image.get_xarray_dask_stack()
        if 'S' not in xd.dims:
            return xd.isel(I=0, T=t, C=c, Z=z)
        return xd.isel(I=0, S=0, T=t, C=c, Z=z)


    @property
    def sizeT(self):
        return self._aics_image.dims.T

    @property
    def sizeZ(self):
        return self._aics_image.dims.Z

    @property
    def sizeC(self):
        return self._aics_image.dims.C

    @property
    def sizeX(self):
        return self._aics_image.dims.X

    @property
    def sizeY(self):
        return self._aics_image.dims.Y

    def as_texture(self, c=0, z=0, t=0, **kwargs):
        if self._memmap is not None:
            return self._memmap.as_texture(c=c, z=z, t=t, **kwargs)
        img = self.image(c=c, z=z, t=t).values
        img = img / np.max(img)
        texture = np.dstack([img, img, img, np.ones((self._aics_image.dims.Y, self._aics_image.dims.X))])
        return self._aics_image.dims.X, self._aics_image.dims.Y, 4, texture.flatten()

    @property
    def shape(self):
        """
        The shape of the image data
        :return: the 5D array shape
        :rtype: tuple of int
        """
        return self._aics_image.shape

    def compute_drift(self, z=0, c=0, max_mem=5000):
        if self._memmap is not None:
            return self._memmap.compute_drift(z=z, c=c, max_mem=max_mem)
        stabilizer = VidStab()
        for frame in range(0, self.sizeT):
            _ = stabilizer.stabilize_frame(
                input_frame=np.uint8(np.array(self.image(t=frame, z=z, c=c)) / 65535 * 255), smoothing_window=1)
        return pd.DataFrame(stabilizer.transforms, columns=('dx', 'dy', 'dr')).cumsum(axis=0)

    def correct_drift(self, drift, filename=None, max_mem=5000):
        new_memmap = MemMapTiff(filename, ome=True, metadata={'axes': 'CTZYX'},
                                shape=(self.sizeC, self.sizeT, self.sizeZ, self.sizeY, self.sizeX), dtype=np.uint16)
        if self._memmap is not None:
            self._memmap.correct_drift(new_memmap, drift, max_mem=max_mem)
        else:
            for c in range(0, self.sizeC):
                for z in range(0, self.sizeZ):
                    new_memmap.data[c, 0 , z, ...] = self.image(c=c, t=0, z=z)
                    for idx in drift.index:
                        new_memmap.data[c, idx + 1, z, ...] = cv.warpAffine(np.array(self.image(c=c, t=idx+1, z=z)),
                                                                            np.float32(
                                                                                [[1, 0, -drift.iloc[idx].dx],
                                                                                 [0, 1, -drift.iloc[idx].dy]]),
                                                                            self.image(c=c, t=idx+1, z=z).shape)
                        if psutil.Process().memory_info().rss / (1024 * 1024) > max_mem:
                            self.refresh()


class MemMapTiff:
    """
    A class to access image data stored in a multi-paged TIFF file defined by an explicit path. The file is memory
    mapped to save RAM.
    """

    def __init__(self, path, **kwargs):
        self.path = path
        self._image_data = tifffile.memmap(path, **kwargs)

    @property
    def data(self):
        """
        Property returning image data as a memory mapped numpy array.
        :return: data array
        :rtype: memory mapped numpy array
        """
        if self._image_data._mmap.closed:
            self.open()
        if len(self._image_data.shape) == 4:
            return np.expand_dims(self._image_data, 0)
        if len(self._image_data.shape) == 3:
            return np.expand_dims(np.expand_dims(self._image_data, 0), 0)
        if len(self._image_data.shape) == 2:
            return np.expand_dims(np.expand_dims(np.expand_dims(self._image_data, 0), 0), 0)
        return self._image_data

    def image(self, c=0, z=0, t=0):
        """
        Return a single-channel 2D image (one layer of one frame)
        :param c: channel index
        :type c: int
        :param z: layer index
        :type z: int
        :param t: frame index
        :type t: int
        :return: the 2D image
        :rtype: 2D memmap array
        """
        return self.data[c, t, z, ...]

    def as_texture(self, c=0, z=0, t=0):
        img = self.image(c=c, z=z, t=t)
        img = img / np.max(img)
        width = img.shape[0]
        height = img.shape[1]
        channels = 4
        texture = np.dstack([img, img, img, np.sign(img)])
        return width, height, channels, texture.flatten()

    @property
    def shape(self):
        """
        The shape of the image data
        :return: the 5D array shape
        :rtype: tuple of int
        """
        return self.data.shape

    def open(self):
        """
        Open the memory mapped file to access data
        """
        self._image_data = tifffile.memmap(self.path)

    def close(self):
        """
        Close the memory mapped file
        """
        if not self._image_data._mmap.closed:
            self._image_data._mmap.close()

    def flush(self):
        """
        Flush the data to save changes to the meory mapped file
        """
        if not self._image_data._mmap.closed:
            self._image_data._mmap.flush()

    def refresh(self):
        """
        Close and open the memory mapped file to save memory if needed. Useful when creating a new file or making lots
        of changes.
        """
        self.close()
        self.open()

    def compute_drift(self, z=0, c=0, max_mem=5000):
        """
        Compute x,y drift of frames along time for a channel and a layer. Return the list of dx,dy shifts to be used
        with the correct_drift method
        :param z: the layer index
        :type z: int
        :param c: the channel index
        :type c: int
        :param max_mem: maximum memory in MB to use for caching memmap. If that amount of memory is reached, then the
        memmap is refreshed to clear memory
        :type max_mem: int
        :return: the list of dx,dy shifts
        :rtype: pandas DataFrame with headers dx, dy and dr, frame index is equal to the DataFrame index
        """
        stabilizer = VidStab()

        for frame in range(0, self.shape[1]):
            _ = stabilizer.stabilize_frame(
                input_frame=np.uint8(np.array(self.image(t=frame, z=z, c=c)) / 65535 * 255), smoothing_window=1)
            if psutil.Process().memory_info().rss / (1024 * 1024) > max_mem:
                self.refresh()
        return pd.DataFrame(stabilizer.transforms, columns=('dx', 'dy', 'dr')).cumsum(axis=0)

    def correct_drift(self, new_memmap, drift, max_mem=5000):
        """
        Correct x,y drift of frames along time given a list of dx,dy shifts
        :param drift: the list of dx and dy by frame
        :type drift: pandas DataFrame with headers dx and dy, frame index is equal to the DataFrame index
        :param max_mem: maximum memory in MB to use for caching memmap. If that amount of memory is reached, then the
        memmap is refreshed to clear memory
        :type max_mem: int
        """

        for c in range(0, self.shape[0]):
            for z in range(0, self.shape[2]):
                new_memmap.data[c, 0 , z, ...] = self.data[c, 0, z, ...]
                for idx in drift.index:
                    new_memmap.data[c, idx + 1, z, ...] = cv.warpAffine(np.array(self.data[c, idx + 1, z, ...]), np.float32(
                        [[1, 0, -drift.iloc[idx].dx], [0, 1, -drift.iloc[idx].dy]]),
                                                                  self.data[c, idx + 1, z, ...].shape)
                    if psutil.Process().memory_info().rss / (1024 * 1024) > max_mem:
                        self.refresh()


class MultipleTiff():
    """
    A class to handle image data stored in multiple TIFF files (one per field of view) defined by a list path names.
    """

    def __init__(self, path, pattern=None):
        self._df = pd.DataFrame({'path': path})
        new_cols = {'C': 0, 'T': 0, 'Z': 0}
        new_cols.update(self._df.apply(lambda x: re.search(pattern, x['path']).groupdict(),
                                       axis=1, result_type='expand').to_dict())
        self._df = self._df.merge(pd.DataFrame(new_cols), left_index=True, right_index=True)
        stacks = self._df.sort_values(by=['C', 'T', 'Z']).groupby(['C', 'T'])
        file_groups = {s: stacks.get_group(s).path.tolist() for s in stacks.groups}

        dims = {dim: self._df.sort_values(by=['C', 'T', 'Z'])[dim].drop_duplicates().to_list() for dim in
                ['C', 'T', 'Z']}

        img_res = TiffFile(self._df.loc[0, 'path']).asarray().shape
        self._shape = (len(dims['C']), len(dims['T']), len(dims['Z']), img_res[0], img_res[1],)

        self._files = []
        for c, channel in enumerate(dims['C']):
            self._files.append([])
            for t, frame in enumerate(dims['T']):
                self._files[c].append([])
                for z, _ in enumerate(dims['Z']):
                    self._files[c][t].append(file_groups[(channel, frame)][z])
        self._files = np.array(self._files)

    def tiff_file(self, c=0, t=0, z=0):
        """
        Return the file corresponding to the image specified by the c, t and z coordinates
        :param c: the channel index
        :type c: int
        :param t: the time index
        :type t: int
        :param z: the layer index
        :type z: int
        :return: the file specified by c, t, z
        :rtype: TiffFile
        """
        return TiffFile(self._files[c, t, z])

    def image(self, c=0, t=0, z=0):
        """
        Return image data (2D) corresponding to the file specified by the c, t, and z coordinates
        :param c: the channel index
        :type c: int
        :param t: the time index
        :type t: int
        :param z: the layer index
        :type z: int
        :return: the image data
        :rtype: ndarray
        """
        tiff_file = self.tiff_file(c, t, z)
        img = tiff_file.asarray()
        tiff_file.close()
        return img

    @property
    def shape(self):
        """
        The shape of the image data
        :return: the 5D array shape
        :rtype: tuple of int
        """
        return self._shape

    def as_memmap_tiff(self, filename, max_mem=500):
        """
        Convert the set of files to a single multi-paged TIFF file.
        :param filename: the name of the target multi-paged file
        :type filename: str
        :param max_mem: memory limit before refreshing the target multi-paged file during its creation
        :type max_mem: int
        :return: the target multi-paged TIFF file
        :rtype: MemMapTiff
        """
        single_tiff = MemMapTiff(filename, ome=True, metadata={'axes': 'CTZYX'},
                                 shape=self._shape, dtype=np.uint16)
        for c, _ in enumerate(self._files):
            for t, _ in enumerate(self._files[c]):
                single_tiff.data[c, t, ...] = TiffSequence(self._files[c, t, :]).asarray()
                if psutil.Process().memory_info().rss / (1024 * 1024) > max_mem:
                    single_tiff.refresh()
        return single_tiff

    def compute_drift(self, z=0, c=0, ):
        """
        Compute x,y drift of frames along time for a channel and a layer. Return the list of dx,dy shifts to be used
        with the correct_drift method
        :param z: the layer index
        :type z: int
        :param c: the channel index
        :type c: int
        :return: the list of dx,dy shifts
        :rtype: pandas DataFrame with headers dx, dy and dr, frame index is equal to the DataFrame index
        """
        stabilizer = VidStab()

        for frame in range(0, self.shape[1]):
            _ = stabilizer.stabilize_frame(
                input_frame=np.uint8(np.array(self.image(t=frame, z=z, c=c)) / 65535 * 255), smoothing_window=1)
        return pd.DataFrame(stabilizer.transforms, columns=('dx', 'dy', 'dr')).cumsum(axis=0)

    def correct_drift(self, drift):
        """
        Correct x,y drift of frames along time given a list of dx,dy shifts
        :param drift: the list of dx and dy by frame
        :type drift: pandas DataFrame with headers dx and dy, frame index is equal to the DataFrame index
        """
        for idx in drift.index:
            for c in range(0, self.shape[0]):
                for z in range(0, self.shape[2]):
                    tifffile.imwrite(self._files[c, idx + 1, z],
                                     cv.warpAffine(np.array(self.image(c=c, t=idx + 1, z=z)), np.float32(
                                         [[1, 0, -drift.iloc[idx].dx], [0, 1, -drift.iloc[idx].dy]]),
                                                   self.image(c=c, t=idx + 1, z=z).shape))


class ImageResourceDeprecated:
    """
    A class to access image data stored in files defined by an explicit path, a path pattern or a list of paths. Image
    data can be loaded from these files as a 5D matrix with the following arbitrary order: t, z, x, y, c
    Frames, layers or channels can be added to the data, using the corresponding methods.
    """

    def __init__(self, path, t_pattern=None):
        self.path = path
        self.data = np.array([])
        if isinstance(self.path, str):
            self.path = glob.glob(path)
        self.t_pattern = t_pattern

    def load_data(self):
        """
        Load the image data from files as a 5D matrix
        """
        print('Loading data...')
        if self.data.size == 0:
            if self.t_pattern is None:
                self.data = ImageResourceDeprecated.get_data_from_path(self.path)
            else:
                r_t_comp = re.compile(self.t_pattern)
                path_str = ' '.join(sorted(self.path))
                for t in sorted(set(r_t_comp.findall(path_str))):
                    print(sorted(glob.glob(f'{t}*')))
                    if self.data.size == 0:
                        self.data = ImageResourceDeprecated.get_data_from_path(sorted(glob.glob(f'{t}*')))
                    else:
                        self.stack(ImageResourceDeprecated.get_data_from_path(sorted(glob.glob(f'{t}*'))))
        return self

    @staticmethod
    def get_data_from_path(path_list):
        """
        Get image data as a 5D matrix resulting from the concatenation of images in a collection defined by a list of
        files
        :param path_list: the list of image files
        :type path_list: list of str
        :return: the image data array
        :rtype: numpy ndarray
        """
        data = np.array([skio.ImageCollection(p).concatenate() for p in path_list])
        if len(data.shape) == 4:
            data = np.expand_dims(data, -1)
        return data

    @property
    def shape(self):
        """
        Returns the shape of data if data has been loaded, otherwise it temporarily loads the data to determine its
        shape before liberating the memory.
        :return: the shape of data
        :rtype: tuple
        """
        if self.data.size == 0:
            shape = self.load_data().shape
            self.free_data()
            return shape
        return self.data.shape

    def free_data(self):
        """
        Liberates previously loaded data to free memory.
        """
        self.data = np.array([])

    def add_frames(self, data):
        """
        Add new frames to the image data (first dimension)
        :param data: new frames having the same dimensions as the current image data
        :type data: a 5D ndarray
        :return: the current object
        """
        self.data = np.concatenate([self.data, data])
        return self

    def stack(self, data):
        """
        Add new layers to the image data (second dimension)
        :param data: new layers having the same dimensions as the current image data
        :type data: a 5D ndarray
        :return: the current object
        """
        self.data = np.hstack([self.data, data])
        return self

    @property
    def z_max(self):
        """
        Property returning the maximum value across all z layers at each pixel for each unit of time and each channel
        :return: the resulting 5D matrix with only one layer
        :rtype: 5D numpy array
        """
        data = np.amax(self.data, axis=1)
        if len(data.shape) == 4:
            data = np.expand_dims(data, 1)
        return data

    def add_channels(self, data):
        """
        Add new channels to the image data (last dimension)
        :param data: new channels having the same dimensions as the current image data
        :type data: a 5D ndarray
        :return: the current object
        """
        self.data = np.block([self.data, data])
        return self

    @staticmethod
    def compose(data_list):
        """
        Compose a multi-channel image from a list of single channel images.
        There is no verification that input data are actually single channels, so this method may also be used to
        create matrices that are not really multi-channel images but an arbitrary set of grayscale images stored in
        the channel dimension
        :param data_list: the 5D image data to compose
        :type data_list: a list of 5D ndarrays
        :return: the composite multi-channel image
        :rtype: a ndarray
        """
        return np.block(data_list)


class HDF5dataset(ImageResourceDeprecated):
    """
    A class to access image data stored in datasets of a HDF5 file. Image data can be loaded from these datasets as a
    5D matrix with the following arbitrary order: t, z, x, y, c
    Frames, layers or channels can be added to the data, using the corresponding methods of the parent class.
    """

    def __init__(self, path, dataset=None):
        super().__init__(path)
        self.path = path
        self.h5_file = h5py.File(self.path)
        self.ds_names = dataset if isinstance(dataset, list) else [dataset]

    def load_data(self):
        """
        Load the image data from a dataset in a HDF5 file as a 5D matrix
        """
        self.ds_names = self.h5_file.keys() if self.ds_names is None else self.ds_names
        self.data = np.concatenate([[self.h5_file[ds_name] for ds_name in self.ds_names]])

    @property
    def dataset(self):
        """
        dataset property to access the list of datasets associated with the data to load
        :return: a list of dataset names
        """
        return self.ds_names

    @dataset.setter
    def dataset(self, dataset=None):
        self.ds_names = dataset if isinstance(dataset, list) else [dataset]

    def close(self):
        """
        Close the HDF5 file
        """
        self.h5_file.close()
