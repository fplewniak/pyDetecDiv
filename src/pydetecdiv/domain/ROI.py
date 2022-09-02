#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from pydetecdiv.domain.dso import NamedDSO
from pydetecdiv.domain.FOV import FOV


class ROI(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of Regions of interest (ROI)
    """

    @property
    def fov(self):
        """
        property returning the FOV object this ROI is a region of
        :return: the parent FOV object
        """
        return self.project.get_object_by_id(FOV, self.data['fov'])

    @fov.setter
    def fov(self, fov: FOV = None):
        self.data['fov'] = fov.id

    @property
    def top_left(self):
        """
        The top-left corner of the ROI in the FOV
        :return: a tuple of two int with the coordinates of the top-left corner
        """
        return self.data['x0'], self.data['y0']

    @top_left.setter
    def top_left(self, top_left: tuple = (0, 0)):
        # TODO check the values are within the FOV image
        (self.data['x0'], self.data['y0']) = top_left

    @property
    def bottom_right(self):
        """
        The bottom-right corner of the ROI in the FOV
        :return: a tuple of two int with the coordinates of the bottom-right corner
        """
        return self.data['x1'], self.data['y1']

    @bottom_right.setter
    def bottom_right(self, bottom_right: tuple = (-1, -1)):
        # TODO check the values are within the FOV image
        (self.data['x1'], self.data['y1']) = bottom_right

    @property
    def shape(self):
        x1 = self.fov.shape[0] - 1 if self.bottom_right[0] < 0 else self.bottom_right[0]
        y1 = self.fov.shape[1] - 1 if self.bottom_right[1] < 0 else self.bottom_right[1]
        shape = (x1 - self.top_left[0] + 1, y1 - self.top_left[1] + 1)
        return shape

    def __repr__(self):
        return f'{self.id} {self.name} {self.top_left} {self.bottom_right}'

    def __dict__(self):
        return dict(id=self.id, name=self.name, fov=dict(id=self.fov.id, name=self.fov.name), top_left=self.top_left,
                    bottom_right=self.bottom_right)
