#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from pydetecdiv.exceptions import JuttingError
from pydetecdiv.domain.dso import NamedDSO, BoxedDSO, DsoWithImageData
from pydetecdiv.domain.FOV import FOV


class ROI(NamedDSO, BoxedDSO, DsoWithImageData):
    """
    A business-logic class defining valid operations and attributes of Regions of interest (ROI)
    """

    def __init__(self, fov=None, **kwargs):
        super().__init__(**kwargs)
        self._fov = fov if isinstance(fov, FOV) or fov is None else self.project.get_object('FOV', fov)
        self.validate(updated=False)

    def delete(self):
        """
        Deletes this ROI if and only if it is not the full-FOV one which should serve to keep track of original data.
        """
        if self is not self.fov.full_fov_roi:
            self.project.delete(self)

    def check_validity(self):
        """
        Checks the current ROI lies within its parent. If it does not, this method will throw a JuttingError exception
        """
        if not self.box.lies_in(self.fov.box):
            raise JuttingError(self, self.fov)

    @property
    def fov(self):
        """
        property returning the FOV object this ROI is a region of
        :return: the parent FOV object
        :rtype: FOV
        """
        return self._fov

    @fov.setter
    def fov(self, fov):
        self._fov = fov if isinstance(fov, FOV) or fov is None else self.project.get_object('FOV', fov)
        self.validate()

    @property
    def bottom_right(self):
        """
        The bottom-right corner of the ROI in the FOV
        :return: the coordinates of the bottom-right corner
        :rtype: a tuple of two int
        """
        return (self.fov.size[0] - 1 if self._bottom_right[0] == -1 else self._bottom_right[0],
                self.fov.size[1] - 1 if self._bottom_right[1] == -1 else self._bottom_right[1])

    @bottom_right.setter
    def bottom_right(self, bottom_right):
        self._bottom_right = bottom_right
        self.validate()

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
            'name': self.name,
            'fov': self._fov.id_,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
            'size': self.size
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    def __repr__(self):
        return f'{self.record()}'
