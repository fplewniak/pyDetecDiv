#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
import numpy as np
from pydetecdiv.domain.dso import DomainSpecificObject


class FileResource(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of a File resource
    """

    def __init__(self, locator=None, mimetype=None, **kwargs):
        super().__init__(**kwargs)
        self.locator = locator
        self.mimetype = mimetype
        self.validate(updated=False)

    def check_validity(self):
        """
        Checks the current FOV is valid
        """
        ...

    def record(self, no_id=False):
        """
        Returns a record dictionary of the current FOV, including or not a list of associated ROIs as a sub-dictionary
        :param include_roi_list: if True, the record will contain a field 'roi_list' with the associated ROIs
        :type include_roi_list: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'locator': self.locator,
            'mimetype': self.mimetype,
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    def __repr__(self):
        return f'{self.record()}'

    @property
    def image_data(self):
        """
        Returns the list of ImageData objects whose parent is the current File resource
        :return: the list of associated Image data
        :rtype: list of ImageData objects
        """
        return self.project.get_linked_objects('ImageData', to=self)

    @property
    def fov_list(self):
        """
        Returns the list of FOV objects linked to the current File resource. This link is transitive and is obtained
        through the link to ImageData
        :return: the list of associated FOV
        :rtype: list of FOV objects
        """
        return list(np.array([img.fov_list for img in self.image_data]).flat)
