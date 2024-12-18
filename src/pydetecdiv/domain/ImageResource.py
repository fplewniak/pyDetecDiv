#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
from typing import TYPE_CHECKING, TypeVar, Any

if TYPE_CHECKING:
    from pydetecdiv.domain.Data import Data

import os

import numpy as np
import pandas
from PIL import Image
from aicsimageio.dimensions import Dimensions

from pydetecdiv.domain.MultiFileImageResource import MultiFileImageResource
from pydetecdiv.domain.SingleFileImageResource import SingleFileImageResource
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.domain.Dataset import Dataset
from pydetecdiv.domain.ImageResourceData import ImageResourceData
from pydetecdiv.domain.Hdf5ImageResource import Hdf5ImageResource
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.settings import get_config_value


class ImageResource(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of Image resources
    """

    def __init__(self, dataset: int | Dataset, fov: int | FOV, multi: bool,
                 xdim: int = -1, ydim: int = -1, zdim: int = -1, cdim: int = -1, tdim: int = -1,
                 xyscale: float = 1, tscale: float = 1, zscale: float = 1,
                 xyunit: float = 1e-6, zunit: float = 1e-6, tunit: float = 1e-3, key_val: dict = None,
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

        self._image_files_5d = None
        self._image_files = None
        self.pattern = self._pattern
        # self.fov = self._fov

    @property
    def dataset(self) -> Dataset:
        """
        the dataset corresponding to this image resource
        """
        return self.project.get_object('Dataset', self._dataset)

    @property
    def fov(self) -> FOV:
        """
        the FOV corresponding to this image resource
        """
        return self.project.get_object('FOV', self.fov_id)

    @property
    def drift(self) -> pandas.DataFrame | None:
        if self.key_val is not None and 'drift' in self.key_val:
            drift_path = os.path.join(get_config_value('project', 'workspace'),
                                      self.fov.project.dbname, self.key_val['drift'])
            return pandas.read_csv(drift_path)
        return None

    @property
    def drift_method(self) -> str | None:
        if self.key_val is not None and 'drift_method' in self.key_val:
            return self.key_val['drift_method']
        return None

    @property
    def data_list(self) -> list['Data']:
        """
        Property returning the Data objects associated to this FOV

        :return: the data associated to this FOV
        :rtype: list of Data objects
        """
        return self.project.get_linked_objects('Data', to=self)

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """
        The image resource shape (should habitually be 5D with the following dimensions TCZYX)
        """
        return (self.tdim, self.cdim, self.zdim, self.ydim, self.xdim)

    @property
    def dims(self) -> Dimensions:
        """
        The image resource dimensions with their size
        """
        return Dimensions("TCZYX", (self.tdim, self.cdim, self.zdim, self.ydim, self.xdim))

    @property
    def xdim(self) -> int:
        """
        The image resource X dimension size determined from file
        """
        if (self._xdim == -1) and len(self.project.get_linked_objects('Data', self)):
            self._ydim, self._xdim = self.set_image_shape_from_file()
        return self._xdim

    @xdim.setter
    def xdim(self, xdim: int) -> None:
        self._xdim = xdim

    @property
    def ydim(self) -> int:
        """
        The image resource Y dimension size determined from file
        """
        if (self._ydim == -1) and len(self.project.get_linked_objects('Data', self)):
            self._ydim, self._xdim = self.set_image_shape_from_file()
        return self._ydim

    @ydim.setter
    def ydim(self, ydim: int) -> None:
        self._ydim = ydim

    @property
    def tdim(self) -> int:
        """
        The image resource Y dimension size determined from file
        """
        if (self._tdim == -1) and len(self.project.get_linked_objects('Data', self)):
            self._tdim = self.image_resource_data().sizeT
        return self._tdim

    @tdim.setter
    def tdim(self, tdim: int) -> None:
        self._tdim = tdim

    @property
    def cdim(self) -> int:
        """
        The image resource Y dimension size determined from file
        """
        if (self._cdim == -1) and len(self.project.get_linked_objects('Data', self)):
            self._cdim = self.image_resource_data().sizeC
        return self._cdim

    @cdim.setter
    def cdim(self, cdim: int) -> None:
        self._cdim = cdim

    @property
    def zdim(self) -> int:
        """
        The image resource Y dimension size determined from file
        """
        if (self._zdim == -1) and len(self.project.get_linked_objects('Data', self)):
            self._zdim = self.image_resource_data().sizeZ
        return self._zdim

    @zdim.setter
    def zdim(self, zdim: int) -> None:
        self._zdim = zdim

    def set_image_shape_from_file(self) -> tuple[int, int]:
        """
        The image shape determined from first file
        """
        # with Image.open(self.project.get_linked_objects('Data', self)[0].url) as img:
        with Image.open(self.image_files[0]) as img:
            self._xdim, self._ydim = img.size
        self.project.save(self)
        return self._ydim, self._xdim

    @property
    def sizeT(self) -> int:
        """
        The number of frames
        """
        return self.dims.T

    @property
    def sizeC(self) -> int:
        """
        the number of channels
        :return:
        """
        return self.dims.C

    @property
    def sizeZ(self) -> int:
        """
        the number of layers
        :return:
        """
        return self.dims.Z

    @property
    def sizeY(self) -> int:
        """
        the height of the images
        """
        return self.dims.Y

    @property
    def sizeX(self) -> int:
        """
        the width of the images
        """
        return self.dims.X

    def image_resource_data(self) -> ImageResourceData:
        """
        Creates a ImageResourceData object with the appropriate sub-class according to the multi parameter
        :return: the ImageResourceData object
        :rtype: ImageResourceData (SingleFileImageResource or MultiFileImageResource)
        """
        if self.key_val is not None and 'hdf5' in self.key_val:
            return Hdf5ImageResource(image_resource=self)
        if not self.multi:
            return SingleFileImageResource(image_resource=self)
        return MultiFileImageResource(image_resource=self)

    @property
    def image_files_5d(self) -> np.ndarray[str] | None:
        """
        property returning the list of file paths as a 3D array. Each file contains a XY 2D image, and there is one
        file for each T, C,Z combination of coordinates
        :return: 3D array of file paths
        :rtype: array of str
        """
        if self._image_files_5d is None:
            data_list = self.project.get_linked_objects('Data', self)

            if self.multi:
                self._image_files_5d = np.empty((self.sizeT, self.sizeC, self.sizeZ), dtype=object)
                for data in sorted(data_list, key=lambda x: (x.t, x.c, x.z)):
                    self._image_files_5d[data.t, data.c, data.z] = data.url
            else:
                self._image_files_5d = None
        return self._image_files_5d

    @property
    def image_files(self) -> list[str]:
        """
        property returning the list of all files associated with this image resource
        :return: list of image files
        :rtype: list of str (file paths)
        """
        if self._image_files is None:
            self._image_files = [d.url for d in
                                 sorted(self.project.get_linked_objects('Data', self), key=lambda x: (x.t, x.c, x.z))]
        return self._image_files

    @property
    def _pattern(self) -> str:
        """
        property returning the pattern defining the dimensions for the dataset associated with this image resource
        :return: pattern
        :rtype: regex str
        """
        return self.dataset.pattern

    def record(self, no_id: bool = False) -> dict[str, Any]:
        """
        Returns a record dictionary of the current Image resource

        :param no_id: if True, does not return id_ (useful for transferring from one project to another)
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'dataset': self._dataset,
            'shape'  : (self.tdim, self.cdim, self.zdim, self.ydim, self.xdim),
            'xyscale': self.xyscale,
            'xyunit' : self.xyunit,
            'zscale' : self.zscale,
            'zunit'  : self.zunit,
            'tscale' : self.tscale,
            'tunit'  : self.tunit,
            'fov'    : self.fov_id,
            'multi'  : self.multi,
            'uuid'   : self.uuid,
            'key_val': self.key_val,
            }
        if not no_id:
            record['id_'] = self.id_
        return record
