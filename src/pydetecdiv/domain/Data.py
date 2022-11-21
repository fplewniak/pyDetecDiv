#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from collections import namedtuple
from pandas import DataFrame
from pydetecdiv.domain.dso import NamedDSO
from pydetecdiv.domain.ROI import ROI


class Data(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of 5D Image data related to ROIs
    """

    def __init__(self, uuid, dataset, author, date, url, format_, source_dir, meta_data, key_val, **kwargs):
        super().__init__(**kwargs)
        self.uuid = uuid
        self.dataset = dataset
        self.author = author
        self.date = date
        self.url = url
        self.format_ = format_
        self.source_dir = source_dir
        self.meta_data = meta_data
        self.key_val = key_val
        self.validate(updated=False)

    def check_validity(self):
        """
        Checks the current ImageData object is valid and consistent with its image content. If
        not, the shape values are updated
        """
        ...

    def validate(self, updated=True):
        """
        Validates the current FOV and checks whether it is associated with an initial ROI. If not, the initial ROI is
        created
        :param updated: True if the FOV has been updated, False otherwise
        :type updated: bool
        """
        super().validate(updated)

    @property
    def fov_list(self):
        """
        Returns the list of ROI objects associated to the current Image data
        :return: the list of associated ROIs
        :rtype: list of ROI objects
        """
        return self.project.get_linked_objects('FOV', to=self)


    def record(self, no_id=False):
        """
        Returns a record dictionary of the current ROI
        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to
        another
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'uuid': self.uuid,
            'name': self.name,
            'dataset': self.dataset,
            'author': self.author,
            'date': self.date,
            'url': self.url,
            'format_': self.format_,
            'source_dir': self.source_dir,
            'meta_data': self.meta_data,
            'key_val': self.key_val
        }
        if not no_id:
            record['id_'] = self.id_
        return record
