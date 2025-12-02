#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydetecdiv.domain.Entity import Entity

from typing import Any

from pydetecdiv.exceptions import JuttingError
from pydetecdiv.domain.dso import NamedDSO, BoxedDSO
from pydetecdiv.domain.FOV import FOV


class ROI(NamedDSO, BoxedDSO):
    """
    A business-logic class defining valid operations and attributes of Regions of interest (ROI)
    """

    def __init__(self, fov: int | FOV = None, key_val: dict[str, Any] = None,**kwargs):
        super().__init__(**kwargs)
        self._fov = fov.id_ if isinstance(fov, FOV) else fov
        self.key_val = key_val
        if self.id_ is None:
            self.fov.set_timestamp()
        self.validate(updated=False)

    def delete(self) -> None:
        """
        Deletes this ROI if and only if it is not the full-FOV one which should serve to keep track of original data.
        """
        if self is not self.fov.initial_roi:
            self.fov.set_timestamp()
            self.project.delete(self)

    # def check_validity(self) -> None:
    #     """
    #     Checks the current ROI lies within its parent. If it does not, this method will throw a JuttingError exception
    #     """
    #     # if not self.box.lies_in(self.fov.box):
    #     #     raise JuttingError(self, self.fov)
    #     ...

    @property
    def entities(self) -> list['Entity']:
        return self.project.get_linked_objects('Entity', to=self)

    @property
    def fov(self) -> FOV:
        """
        property returning the FOV object this ROI is a region of

        :return: the parent FOV object
        :rtype: FOV
        """
        return self.project.get_object('FOV', self._fov)

    @fov.setter
    def fov(self, fov: FOV) -> None:
        self._fov = fov.id_ if isinstance(fov, FOV) else fov
        self.validate()

    @property
    def sizeT(self) -> int:
        return self.fov.sizeT

    @property
    def bottom_right(self) -> tuple[int, int]:
        """
        The bottom-right corner of the ROI in the FOV

        :return: the coordinates of the bottom-right corner
        :rtype: a tuple of two int
        """
        return (self.fov.size[0] - 1 if self._bottom_right[0] == -1 else self._bottom_right[0],
                self.fov.size[1] - 1 if self._bottom_right[1] == -1 else self._bottom_right[1])

    @bottom_right.setter
    def bottom_right(self, bottom_right: tuple[int, int]) -> None:
        self._bottom_right = bottom_right
        self.validate()

    def record(self, no_id: bool = False) -> dict[str, Any]:
        """
        Returns a record dictionary of the current ROI

        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to another
        :type no_id: bool
        :return: record dictionary
        """
        record = {
            'name': self.name,
            'fov': self._fov,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
            'size': self.size,
            'uuid': self.uuid,
            'key_val': self.key_val,
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    @property
    def info(self) -> str:
        return f"""
Name:                 {self.name}
FOV:                  {self.fov.name}
Position:             {self.top_left} - {self.bottom_right}
Size:                 {self.size}
number of datasets:   {len(self.project.get_linked_objects('Dataset', to=self))}
number of data files: {len(self.project.get_linked_objects('Data', to=self))}
        """
