#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
import json
import os

from pydetecdiv.domain.dso import NamedDSO
from pydetecdiv.settings import get_config_value


class Data(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of data
    """

    def __init__(self, uuid, dataset, author, date, url, format_, source_dir, meta_data, key_val, **kwargs):
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
        self.validate(updated=False)

    @property
    def dataset(self):
        """
        Property returning the Dataset object this data belongs to
        :return: the Dataset this Data belongs to
        :rtype: Dataset object
        """
        return self.project.get_object('Dataset', uuid=self.dataset_)

    @property
    def url(self):
        if self.url_.startswith('/'):
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
            'format_': self.format_,
            'source_dir': self.source_dir,
            'meta_data': self.meta_data,
            'key_val': self.key_val,
            'uuid': self.uuid
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
