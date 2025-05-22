#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to points
"""
from typing import Any

from PySide6.QtWidgets import QGraphicsEllipseItem

from pydetecdiv.domain.Entity import Entity
from pydetecdiv.domain.dso import NamedDSO


class Point(NamedDSO):
    """
    A class defining a point with its properties and available methods
    """

    def __init__(self, point: QGraphicsEllipseItem = None, label: int = 1, frame: int = None, entity: Entity = None,
                 x: float = 0, y: float = 0, key_val: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.graphics_item = point
        self.label = label
        self.frame = frame
        self._entity = entity.id_ if isinstance(entity, Entity) else entity
        self._x = x
        self._y = y
        self.key_val = key_val
        self.validate(updated=False)

    @property
    def object(self):
        """
        property returning the Entity object this point corresponds to. (this is a legacy method expected to be removed
        not to be used)
        """
        return self.entity

    @property
    def entity(self) -> Entity:
        """
        property returning the ROI object this entity belongs to

        :return: the parent ROI object
        """
        return self.project.get_object('Entity', self._entity)

    def change_point(self, point: QGraphicsEllipseItem) -> None:
        """
        changes the specified bounding box
        :param box: the new graphics item for the bounding box
        """
        self.graphics_item = point
        if point is None:
            self.name = None
        else:
            self.name = point.data(0)

    @property
    def x(self) -> float | None:
        """
        the x coordinate of the point
        """
        if self.graphics_item is None:
            self.graphics_item = QGraphicsEllipseItem(0, 0, 2, 2)
            self.graphics_item.setPos(self._x, self._y)
        return self.graphics_item.pos().x()

    @property
    def y(self) -> float | None:
        """
        the y coordinate of the point
        """
        if self.graphics_item is None:
            self.graphics_item = QGraphicsEllipseItem(0, 0, 2, 2)
            self.graphics_item.setPos(self._x, self._y)
        return self.graphics_item.pos().y()

    @property
    def coords(self) -> list[float]:
        """
        the coordinates of the point
        """
        if self.x is None:
            return []
        return [self.x, self.y]

    def __repr__(self):
        """
        returns a representation of the point (name and coordinates)
        """
        return f'{self.name=}, {self.coords=}'

    def record(self, no_id: bool = False) -> dict[str, Any]:
        """
        Returns a record dictionary of the current Entity

        :param no_id: if True, the id\_ is not passed included in the record to allow transfer from one project to another
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'uuid'   : self.uuid,
            'entity' : self._entity,
            'name'   : self.name,
            'frame'  : self.frame,
            'x'      : self.x,
            'y'      : self.y,
            'label'  : self.label,
            'key_val': self.key_val,
            }
        if not no_id:
            record['id_'] = self.id_
        return record
