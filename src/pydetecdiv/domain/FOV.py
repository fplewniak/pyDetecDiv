#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
from pydetecdiv.domain.dso import NamedDSO, ImageAssociatedDSO


class FOV(NamedDSO, ImageAssociatedDSO):
    """
    A business-logic class defining valid operations and attributes of Fields of view (FOV)
    """

    def __init__(self, comments: str = None, **kwargs):
        super().__init__(**kwargs)
        self.comments = comments
        self.validate()

    def record(self, include_roi_list=False):
        """
        Returns a record dictionary of the current FOV, including or not a list of associated ROIs as a sub-dictionary
        :param include_roi_list: if True, the record will contain a field 'roi_list' with the associated ROIs
        :type include_roi_list: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'id': self.id,
            'name': self.name,
            'comments': self.comments,
            'shape': self.shape
        }
        if include_roi_list:
            record['roi_list'] = self.roi_list
        return record

    def __repr__(self):
        return f'{self.record()}'

    def __eq__(self, other):
        """
        Defines equality of FOV objects as having the same id, same name and same shape
        :param other: the other FOV object to compare with the current one
        :type other: FOV
        :return: True if both FOVs are equal
        :rtype: bool
        """
        is_eq = [self.id == other.id, self.name == other.name, self.shape == self.shape]
        return all(is_eq)

    @property
    def roi_list(self):
        """
        Returns the list of ROI objects whose parent if the current FOV
        :return: the list of associated ROIs
        :rtype: list of ROI objects
        """
        return self.project.get_roi_list_in_fov(self)
