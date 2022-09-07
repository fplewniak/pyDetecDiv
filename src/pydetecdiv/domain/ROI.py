#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from pydetecdiv.exceptions import JuttingError
from pydetecdiv.domain.dso import NamedDSO, ImageAssociatedDSO
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.utils.Shapes import Box


class ROI(NamedDSO, ImageAssociatedDSO):
    """
    A business-logic class defining valid operations and attributes of Regions of interest (ROI)
    """

    def __init__(self, fov=None, top_left: tuple = (0, 0), bottom_right: tuple = (-1, -1), **kwargs):
        super().__init__(**kwargs)
        self.fov = fov
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.check_validity()

    def __eq__(self, o):
        """
        Defines equality of ROI objects as having the same id, same FOV parent and same location
        :param other: the other ROI object to compare with the current one
        :type other: ROI
        :return: True if both ROIs are equal
        :rtype: bool
        """
        is_eq = [self.id == o.id, self.fov == o.fov, self.top_left == o.top_left, self.bottom_right == o.bottom_right]
        return all(is_eq)

    def check_validity(self):
        """
        Checks the current ROI lies within its parent. If it does not, this method will throw a JuttingError exception
        """
        if not Box(self.top_left, self.bottom_right).lies_in(Box(shape=self.fov.shape)):
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
    def fov(self, fov: FOV = None):
        self._fov = fov if isinstance(fov, FOV) or fov is None else self.project.get_object_by_id(FOV, fov)

    @property
    def top_left(self):
        """
        The top-left corner of the ROI in the FOV
        :return: the coordinates of the top-left corner
        :rtype: a tuple of two int
        """
        return self._top_left

    @top_left.setter
    def top_left(self, top_left: tuple = (0, 0)):
        self._top_left = top_left

    @property
    def bottom_right(self):
        """
        The bottom-right corner of the ROI in the FOV
        :return: the coordinates of the bottom-right corner
        :rtype: a tuple of two int
        """
        return self._bottom_right

    @bottom_right.setter
    def bottom_right(self, bottom_right: tuple = (-1, -1)):
        self._bottom_right = (self.fov.shape[0] - 1 if bottom_right[0] == -1 else bottom_right[0],
                              self.fov.shape[1] - 1 if bottom_right[1] == -1 else bottom_right[1])

    @property
    def shape(self):
        """
        The shape (dimension) of the ROI computed from the coordinates of its corners
        :return: the dimension of the ROI
        :rtype: a tuple of two int
        """
        return self.bottom_right[0] - self.top_left[0] + 1, self.bottom_right[1] - self.top_left[1] + 1

    def record(self):
        """
        Returns a record dictionary of the current ROI
        :return: record dictionary
        :rtype: dict
        """
        return {
            'id': self.id,
            'name': self.name,
            'fov': self._fov,
            'top_left': self._top_left,
            'bottom_right': self._bottom_right,
            'shape': self.shape
        }

    def __repr__(self):
        return f'{self.record()}'
