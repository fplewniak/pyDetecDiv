#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
import os

import numpy as np
import pandas
from PIL import Image
from aicsimageio.dimensions import Dimensions

from pydetecdiv.domain import MultiFileImageResource, SingleFileImageResource, FOV, Dataset
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.settings import get_config_value


class ImageResource(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of Image resources
    """

    def __init__(self, dataset, fov, multi,
                 xdim=-1, ydim=-1, zdim=-1, cdim=-1, tdim=-1,
                 xyscale=1, tscale=1, zscale=1,
                 xyunit=1e-6, zunit=1e-6, tunit=60, key_val=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._dataset = dataset.id_ if isinstance(dataset, Dataset) else dataset
        self.fov_id = fov.id_ if isinstance(fov, FOV) else fov
        self.multi = multi
        self._xdim = xdim
        self._ydim = ydim
        self._zdim = zdim
        self._cdim = cdim
        self._tdim = tdim
        self.xyscale = xyscale
        self.xyunit = xyunit
        self.zscale = zscale
        self.zunit = zunit
        self.tscale = tscale
        self.tunit = tunit
        self.key_val = key_val
        self.validate(updated=False)

        self.image_files_5d = self._image_files_5d
        self.image_files = self._image_files
        self.pattern = self._pattern
        self.fov = self._fov

    @property
    def dataset(self):
        """
        the dataset corresponding to this image resource
        """
        return self.project.get_object('Dataset', self._dataset)

    @property
    def _fov(self):
        """
        the FOV corresponding to this image resource
        """
        return self.project.get_object('FOV', self.fov_id)

    @property
    def drift(self):
        if self.key_val is not None and 'drift' in self.key_val:
            drift_path = os.path.join(get_config_value('project', 'workspace'),
                                  self.fov.project.dbname, self.key_val['drift'])
            return pandas.read_csv(drift_path)
        return None

    @property
    def drift_method(self):
        if self.key_val is not None and 'drift_method' in self.key_val:
            return self.key_val['drift_method']
        return None

    @property
    def data_list(self):
        """
        Property returning the Data objects associated to this FOV

        :return: the data associated to this FOV
        :rtype: list of Data objects
        """
        return self.project.get_linked_objects('Data', to=self)

    @property
    def shape(self):
        """
        The image resource shape (should habitually be 5D with the following dimensions TCZYX)
        """
        return (self.tdim, self.cdim, self.zdim, self.ydim, self.xdim)

    @property
    def dims(self):
        """
        The image resource dimensions with their size
        """
        return Dimensions("TCZYX", (self.tdim, self.cdim, self.zdim, self.ydim, self.xdim))

    @property
    def xdim(self):
        """
        The image resource X dimension size determined from file
        """
        if (self._xdim == -1) and len(self.project.get_linked_objects('Data', self)):
            self._ydim, self._xdim = self.set_image_shape_from_file()
        return self._xdim

    @property
    def ydim(self):
        """
        The image resource Y dimension size determined from file
        """
        if (self._ydim == -1) and len(self.project.get_linked_objects('Data', self)):
            self._ydim, self._xdim = self.set_image_shape_from_file()
        return self._ydim

    @property
    def tdim(self):
        """
        The image resource Y dimension size determined from file
        """
        if (self._tdim == -1) and len(self.project.get_linked_objects('Data', self)):
            self._tdim = self.image_resource_data().sizeT
        return self._tdim

    @tdim.setter
    def tdim(self, tdim):
        self._tdim = tdim

    @property
    def cdim(self):
        """
        The image resource Y dimension size determined from file
        """
        if (self._cdim == -1) and len(self.project.get_linked_objects('Data', self)):
            self._cdim = self.image_resource_data().sizeC
        return self._cdim

    @cdim.setter
    def cdim(self, cdim):
        self._cdim = cdim

    @property
    def zdim(self):
        """
        The image resource Y dimension size determined from file
        """
        if (self._zdim == -1) and len(self.project.get_linked_objects('Data', self)):
            self._zdim = self.image_resource_data().sizeZ
        return self._zdim

    @zdim.setter
    def zdim(self, zdim):
        self._zdim = zdim


    def set_image_shape_from_file(self):
        """
        The image shape determined from first file
        """
        # with Image.open(self.project.get_linked_objects('Data', self)[0].url) as img:
        with Image.open(self._image_files[0]) as img:
            self._xdim, self._ydim = img.size
        self.project.save(self)
        return (self._ydim, self._xdim)

    @property
    def sizeT(self):
        """
        The number of frames
        """
        return self.dims.T

    @property
    def sizeC(self):
        """
        the number of channels
        :return:
        """
        return self.dims.C

    @property
    def sizeZ(self):
        """
        the number of layers
        :return:
        """
        return self.dims.Z

    @property
    def sizeY(self):
        """
        the height of the images
        """
        return self.dims.Y

    @property
    def sizeX(self):
        """
        the width of the images
        """
        return self.dims.X

    def image_resource_data(self):
        """
        Creates a ImageResourceData object with the appropriate sub-class according to the multi parameter
        :return: the ImageResourceData object
        :rtype: ImageResourceData (SingleFileImageResource or MultiFileImageResource)
        """
        if not self.multi:
            return SingleFileImageResource(image_resource=self)
        return MultiFileImageResource(image_resource=self)
        # if not self.multi:
        #     return SingleFileImageResource(self.project.get_linked_objects('Data', self)[0].url,
        #                                      max_mem=5000, fov=self.fov, image_resource=self)
        # return MultiFileImageResource([data.url for data in self.project.get_linked_objects('Data', self)],
        #                          pattern=self.dataset.pattern, max_mem=5000, fov=self.fov, image_resource=self)

    # def image(self, C=0, Z=0, T=0, drift=None):
    #     """
    #     A 2D grayscale image (on frame, one channel and one layer)
    #
    #     :param C: the channel index
    #     :type C: int
    #     :param Z: the layer index
    #     :type Z: int
    #     :param T: the frame index
    #     :type T: int
    #     :return: a 2D data array
    #     :rtype: 2D numpy.array
    #     """
    #     return self.image_files[T, C, Z].image_data()
    # if self._memmap is not None:
    #     s = self.resource.shape
    #     data = np.expand_dims(self._memmap, axis=tuple(i for i in range(len(s)) if s[i] == 1))[T, C, Z, ...]
    # else:
    #     data = self.resource.get_image_dask_data('YX', C=C, Z=Z, T=T).compute()
    # if drift is not None:
    #     return cv.warpAffine(np.array(data),
    #                   np.float32(
    #                       [[1, 0, -drift.dx],
    #                        [0, 1, -drift.dy]]),
    #                   (data.shape[1], data.shape[0]))
    # return data

    @property
    def _image_files_5d(self):
        """
        property returning the list of file paths as a 3D array. Each file contains a XY 2D image, and there is one
        file for each T, C,Z combination of coordinates
        :return: 3D array of file paths
        :rtype: array of str
        """
        data_list = self.project.get_linked_objects('Data', self)
        if self.multi:
            image_files = np.empty((self.sizeT, self.sizeC, self.sizeZ), dtype=object)
            for data in sorted(data_list, key=lambda x: (x.t, x.c, x.z)):
                image_files[data.t, data.c, data.z] = data.url
        else:
            image_files = None
        return image_files

    @property
    def _image_files(self):
        """
        property returning the list of all files associated with this image resource
        :return: list of image files
        :rtype: list of str (file paths)
        """
        return [d.url for d in sorted(self.project.get_linked_objects('Data', self), key=lambda x: (x.t, x.c, x.z))]

    @property
    def _pattern(self):
        """
        property returning the pattern defining the dimensions for the dataset associated with this image resource
        :return: pattern
        :rtype: regex str
        """
        return self.dataset.pattern

    def record(self, no_id=False):
        """
        Returns a record dictionary of the current Image resource

        :param no_id: if True, does not return id_ (useful for transferring from one project to another)
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'dataset': self._dataset,
            'shape': (self.tdim, self.cdim, self.zdim, self.ydim, self.xdim),
            'xyscale': self.xyscale,
            'xyunit': self.xyunit,
            'zscale': self.zscale,
            'zunit': self.zunit,
            'tscale': self.tscale,
            'tunit': self.tunit,
            'fov': self.fov_id,
            'multi': self.multi,
            'uuid': self.uuid,
            'key_val': self.key_val,
        }
        if not no_id:
            record['id_'] = self.id_
        return record
