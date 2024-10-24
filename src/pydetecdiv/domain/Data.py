#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
import json
import os
import tifffile
import numpy as np
from aicsimageio import AICSImage

from pydetecdiv.domain.dso import NamedDSO
from pydetecdiv.settings import get_config_value
from pydetecdiv.domain.ImageResource import ImageResource


class Data(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of data
    """

    def __init__(self, uuid, dataset, author, date, url, format_, source_dir, meta_data, key_val, image_resource,
                 c=None, t=None, z=None, xdim=-1, ydim=-1, **kwargs):
        super().__init__(**kwargs)
        self.uuid = uuid
        self.dataset_ = dataset
        self.author = author
        self.date = date
        self.url_ = url
        self.format_ = format_
        self.source_dir = source_dir
        self.meta_data = meta_data
        self.key_val = key_val
        self._image_resource = image_resource
        self.xdim = xdim
        self.ydim = ydim
        self.c = c
        self.t = t
        self.z = z
        self.validate(updated=False)

    # def image_data(self, T=0, Z=0, C=0):
    #     if self._memmap:
    #         return self._memmap[T, C, Z, ...]
    #     return AICSImage(self.url_).reader.get_image_dask_data('YX').compute()

    @property
    def image_resource(self):
        """
        property returning the Image resource object this Data file is part of

        :return: the parent Image resource object
        :rtype: ImageResource
        """
        return self.project.get_object('ImageResource', self._image_resource)

    @image_resource.setter
    def image_resource(self, image_resource):
        self._image_resource = image_resource.id_ if isinstance(image_resource, ImageResource) else image_resource
        self.validate()

    @property
    def dataset(self):
        """
        Property returning the Dataset object this data belongs to

        :return: the Dataset this Data belongs to
        :rtype: Dataset object
        """
        return self.project.get_object('Dataset', id_=self.dataset_)

    @property
    def url(self):
        """
        URL property of the data file, relative to the workspace directory or absolute path if file are stored in place

        :return: relative or absolute path of the data file
        :rtype: str
        """
        if os.path.isabs(self.url_):
            return self.url_
        return os.path.join(get_config_value('project', 'workspace'), self.project.dbname, self.dataset.name, self.url_)

    @property
    def fov(self):
        """
        Returns the list of FOV objects associated to the current data

        :return: the list of associated FOVs
        :rtype: list of FOV objects
        """
        return self.project.get_linked_objects('FOV', to=self)

    def record(self, no_id=False):
        """
        Returns a record dictionary of the current Data

        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to
        another
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'name': self.name,
            'dataset': self.dataset_,
            'author': self.author,
            'date': self.date,
            'url': self.url,
            'format': self.format_,
            'source_dir': self.source_dir,
            'meta_data': self.meta_data,
            'key_val': self.key_val,
            'uuid': self.uuid,
            'image_resource': self._image_resource,
            'xdim': self.xdim,
            'ydim': self.ydim,
            'z': self.z,
            'c': self.c,
            't': self.t,
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    @property
    def info(self):
        return f"""
Name:             {self.name}
Dataset:          {self.dataset.name} (type: {self.dataset.type_}, run: {self.dataset.run})
FOV:              {self.fov[0].name if len(self.fov) == 1 else len(self.fov)}
Date:             {self.date}
Full path:        {self.url}
Source directory: {self.source_dir}
metadata:         {json.dumps(self.meta_data, indent=4)}
                  {json.dumps(self.key_val, indent=4)}
        """
