#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
import numpy as np
import cv2 as cv
from PIL import Image

from pydetecdiv.domain.ImageResourceData import ImageResourceData
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.domain.Dataset import Dataset
from aicsimageio.dimensions import Dimensions


class ImageResource(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of Image resources
    """

    def __init__(self, dataset, fov, xdim=-1, ydim=-1, zdim=1, cdim=1, tdim=1,
                 xyscale=1, tscale=1, zscale=1,
                 xyunit=1e-6, zunit=1e-6, tunit=60,
                 **kwargs):
        super().__init__(**kwargs)
        self._dataset = dataset.id_ if isinstance(dataset, Dataset) else dataset
        self._fov = fov.id_ if isinstance(fov, FOV) else fov
        self._xdim = xdim
        self._ydim = ydim
        self.zdim = zdim
        self.cdim = cdim
        self.tdim = tdim
        self.xyscale = xyscale
        self.xyunit = xyunit
        self.zscale = zscale
        self.zunit = zunit
        self.tscale = tscale
        self.tunit = tunit
        self.validate(updated=False)

    @property
    def dataset(self):
        return self.project.get_object('Dataset', self._dataset)

    @property
    def fov(self):
        return self.project.get_object('FOV', self._fov)

    @property
    def data_list(self):
        """
        Property returning the Data objects associated to this FOV

        :return: the data associated to this FOV
        :rtype: list of Data objects
        """
        data = self.project.get_linked_objects('Data', to=self)
        return data

    @property
    def shape(self):
        """
        The image resource shape (should habitually be 5D with the following dimensions TCZYX)
        """
        return (self.tdim, self.cdim, self.zdim, self.ydim, self.xdim)

    @property
    def dims(self):
        return Dimensions("TCZYX", (self.tdim, self.cdim, self.zdim, self.ydim, self.xdim))

    @property
    def xdim(self):
        if self._xdim is None and len(self.project.get_linked_objects('Data', self)):
           self._ydim, self._xdim = self.set_image_shape_from_file()
        return self._xdim

    @property
    def ydim(self):
        if self._ydim is None and len(self.project.get_linked_objects('Data', self)):
           self._ydim, self._xdim = self.set_image_shape_from_file()
        return self._ydim

    def set_image_shape_from_file(self):
        # self._ydim, self._xdim = cv.imread(self.project.get_linked_objects('Data', self)[0].url, cv.IMREAD_UNCHANGED).shape
        with Image.open(self.project.get_linked_objects('Data', self)[0].url) as img:
                    self._xdim, self._ydim = img.size
        self.project.save(self)
        return (self._ydim, self._xdim)

    @property
    def sizeT(self):
        return self.dims.T

    @property
    def sizeC(self):
        return self.dims.C

    @property
    def sizeZ(self):
        return self.dims.Z

    @property
    def sizeY(self):
        return self.dims.Y

    @property
    def sizeX(self):
        return self.dims.X

    def image_resource_data(self):
        return ImageResourceData([data.url for data in self.project.get_linked_objects('Data', self)],
                                 pattern=self.dataset.pattern, max_mem=5000, fov=self.fov, image_resource=self)


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
    def image_files(self):
        data_list = self.project.get_linked_objects('Data', self)
        if len(data_list) > 1:
            image_files = np.empty((self.sizeT, self.sizeC, self.sizeZ), dtype=object)
            for data in sorted(data_list, key=lambda x: (x.t, x.c, x.z)):
                image_files[data.t, data.c, data.z] = data
        else:
            image_files = data_list
        return image_files

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
            'fov': self._fov,
            'uuid': self.uuid
        }
        if not no_id:
            record['id_'] = self.id_
        return record
