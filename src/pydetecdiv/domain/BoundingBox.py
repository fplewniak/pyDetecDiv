from typing import Any

from PySide6.QtWidgets import QGraphicsRectItem

from pydetecdiv.domain.Entity import Entity
from pydetecdiv.domain.dso import DomainSpecificObject, NamedDSO


class BoundingBox(NamedDSO):
    """
    A class defining a bounding box with its properties and available methods
    """

    def __init__(self, box: QGraphicsRectItem = None, frame: int = None, entity: Entity = None, key_val: dict = None,
                 x: float = 0, y: float = 0, width: int = 0, height: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.graphics_item = box
        self.frame = frame
        self._entity = entity.id_ if isinstance(entity, Entity) else entity
        self.key_val = key_val
        self.validate(updated=False)
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        if self.graphics_item is None:
            self.graphics_item = QGraphicsRectItem(0, 0, self._width, self._height)
            self.graphics_item.setPos(self._x, self._y)

    @property
    def object(self):
        return self.entity

    @property
    def entity(self) -> Entity:
        """
        property returning the ROI object this entity belongs to

        :return: the parent ROI object
        """
        return self.project.get_object('Entity', self._entity)

    @property
    def x(self) -> float | None:
        """
        the x coordinate of the bounding box (top-left corner)
        """
        if self.graphics_item is None:
            self.graphics_item = QGraphicsRectItem(0, 0, self._width, self._height)
            self.graphics_item.setPos(self._x, self._y)
        return self.graphics_item.pos().x()

    @property
    def y(self) -> float | None:
        """
        the y coordinate of the bounding box (top-left corner)
        """
        if self.graphics_item is None:
            self.graphics_item = QGraphicsRectItem(0, 0, self._width, self._height)
            self.graphics_item.setPos(self._x, self._y)
        return self.graphics_item.pos().y()

    @property
    def width(self) -> int | None:
        """
        the width of the bounding box
        """
        if self.graphics_item is None:
            self.graphics_item = QGraphicsRectItem(0, 0, self._width, self._height)
            self.graphics_item.setPos(self._x, self._y)
        return self.graphics_item.rect().width()

    @property
    def height(self) -> int | None:
        """
        the height of the bounding box
        """
        if self.graphics_item is None:
            self.graphics_item = QGraphicsRectItem(0, 0, self._width, self._height)
            self.graphics_item.setPos(self._x, self._y)
        return self.graphics_item.rect().height()

    @property
    def coords(self) -> list[float]:
        """
        the coordinates of the bounding box (top-left corner / bottom-right corner)
        """
        if self.x is None:
            return []
        return [self.x, self.y, self.x + self.width, self.y + self.height]

    def change_box(self, box: QGraphicsRectItem) -> None:
        """
        changes the specified bounding box
        :param box: the new graphics item for the bounding box
        """
        self.graphics_item = box
        if box is None:
            self.name = None
        else:
            self.name = box.data(0)

    def __repr__(self) -> str:
        """
        returns a representation of the bounding box (name and coordinates)
        """
        return f'Bounding box[{self.name=}, {self.frame=}: {self.coords=}]'

    def record(self, no_id: bool = False) -> dict[str, Any]:
        """
        Returns a record dictionary of the current Entity

        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to another
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
            'width'  : self.width,
            'height' : self.height,
            'key_val': self.key_val,
            }
        if not no_id:
            record['id_'] = self.id_
        return record
