#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 Classes to manipulate Image resources: loading data from files, etc
"""
import glob
import h5py
import numpy as np
import skimage.io as skio


class ImageResource:
    """
    A class to access image data stored in files defined by an explicit path, a path pattern or a list of paths. Image
    data can be loaded from these files as a 5D matrix with the following arbitrary order: t, z, x, y, c
    Frames, layers or channels can be added to the data, using the corresponding methods.
    """

    def __init__(self, path):
        self.path = path
        self.data = np.array([])
        if isinstance(self.path, str):
            self.path = glob.glob(path)

    def load_data(self):
        """
        Load the image data from files as a 5D matrix
        """
        self.data = np.array([skio.ImageCollection(path).concatenate() for path in self.path])
        if len(self.data.shape) == 4:
            self.data = np.expand_dims(self.data, -1)

    def free_data(self):
        """
        Liberates previously loaded data to allow its removal by the garbage collector.
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

    def add_channels(self, data):
        """
        Add new channels to the image data (last dimension)
        :param data: new channels having the same dimensions as the current image data
        :type data: a 5D ndarray
        :return: the current object
        """
        self.data = np.block([self.data, data])
        return self


class HDF5dataset(ImageResource):
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
        :param ds_names: list name of datasets to load
        :type ds_names: list of str
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
