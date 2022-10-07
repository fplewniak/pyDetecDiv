#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 Classes to manipulate Image resources: loading data from files, etc
"""
import glob
import re

import h5py
import numpy as np
import skimage.io as skio


class ImageResource:
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
                self.data = ImageResource.get_data_from_path(self.path)
            else:
                #self.data = self.get_data_from_path(self.path)
                # As a matter of fact, should get a list of lists of files representing time sequences of stacks.
                self.t_pattern = r' (\S+.+\S+) ' if self.t_pattern is None else self.t_pattern
            #     self.z_pattern = r' (\S+.+\S+) ' if self.z_pattern is None else f' (\S+{self.z_pattern}\S+) '
                r_t_comp = re.compile(self.t_pattern)
                path_str = ' '.join(sorted(self.path))
                for t in sorted(set(r_t_comp.findall(path_str))):
                    print(sorted(glob.glob(f'{t}*')))
                    if self.data.size == 0:
                        self.data = ImageResource.get_data_from_path(sorted(glob.glob(f'{t}*')))
                    else:
                        self.stack(ImageResource.get_data_from_path(sorted(glob.glob(f'{t}*'))))
            # r_z_comp = re.compile(self.z_pattern)
            # path_str = ' '.join(sorted(self.path))
            # for t in set(r_t_comp.findall(path_str)):
            #     path_str = ' '.join(sorted(glob.glob(f'{t}*')))
            #     print(r_z_comp.findall(path_str))
            #     if self.data.size == 0:
            #         self.data = self.get_data_from_path(r_z_comp.findall(path_str))
            #     else:
            #         self.stack(self.get_data_from_path(r_z_comp.findall(path_str)))
        return self

    @staticmethod
    def get_data_from_path(path_list):
        data = np.array([skio.ImageCollection(p).concatenate() for p in path_list])
        if len(data.shape) == 4:
            data = np.expand_dims(data, -1)
        return data

    # @property
    # def shape(self):
    #     if self.data.size == 0:
    #         shape = self.load_data().shape
    #         self.free_data()
    #         return shape
    #     return self.data.shape

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
