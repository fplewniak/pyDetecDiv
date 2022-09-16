#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
from pydetecdiv.domain.dso import NamedDSO, BoxedDSO


class FOV(NamedDSO, BoxedDSO):
    """
    A business-logic class defining valid operations and attributes of Fields of view (FOV)
    """

    def __init__(self, comments=None, **kwargs):
        super().__init__(**kwargs)
        self._comments = comments
        self.validate(updated=False)

    def delete(self):
        """
        Delete the current FOV, first deleting all linked ROIs that would be consequently left orphaned
        """
        for roi in self.roi_list:
            roi.delete()
        self.project.delete(self)

    def check_validity(self):
        """
        Checks the current FOV is valid
        """
        ...

    def record(self, include_roi_list=False, no_id = False):
        """
        Returns a record dictionary of the current FOV, including or not a list of associated ROIs as a sub-dictionary
        :param include_roi_list: if True, the record will contain a field 'roi_list' with the associated ROIs
        :type include_roi_list: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'name': self.name,
            'comments': self.comments,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
            'size': self.size
        }
        if not no_id:
            record['id_'] = self.id_
        if include_roi_list:
            record['roi_list'] = self.roi_list
        return record

    def __repr__(self):
        return f'{self.record()}'

    @property
    def comments(self):
        """
        comments property of FOV
        :return: the comments
        :rtype: str
        """
        return self._comments

    @comments.setter
    def comments(self, comments):
        self._comments = comments
        self.validate()

    @property
    def roi_list(self):
        """
        Returns the list of ROI objects whose parent if the current FOV
        :return: the list of associated ROIs
        :rtype: list of ROI objects
        """
        return self.project.get_roi_list_in_fov(self)
