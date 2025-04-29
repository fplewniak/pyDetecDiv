from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QBrush, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsItem, QGraphicsEllipseItem

from pydetecdiv.domain.Entity import Entity
from pydetecdiv.domain.dso import NamedDSO


class Mask(NamedDSO):
    """
    A class defining masks as predicted by SegmentAnything2 from the prompts
    """

    def __init__(self, mask_item: QGraphicsPolygonItem = None, frame: int = None, entity: Entity = None, bin_mask=None,
                 key_val: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.graphics_item = mask_item
        self._ellipse_item = None
        self.frame = frame
        self._entity = entity.id_ if isinstance(entity, Entity) else entity
        self._bin_mask = bin_mask
        self.contour_method = cv2.CHAIN_APPROX_SIMPLE
        self.brush = QBrush()
        self.key_val = key_val
        self.validate(updated=False)

    @property
    def bin_mask(self):
        if self._bin_mask is not None:
            shape = self.entity.roi.size
            return np.frombuffer(self._bin_mask, dtype=bool).reshape((shape[1], shape[0]))
        return None

    @bin_mask.setter
    def bin_mask(self, bin_mask):
        self._bin_mask = bin_mask.tobytes()

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
    def contour(self) -> np.ndarray | None:
        """
        The mask contour as determined according to the specified method
        """
        return self.bitmap2contour(self.bin_mask, self.contour_method)

    @property
    def normalised_contour(self) -> np.ndarray | None:
        """
        The mask contour as determined according to the specified method and relative to ROI size
        """
        contour = self.bitmap2contour(self.bin_mask, self.contour_method)
        shape = self.entity.roi.size
        normalised_contour = np.array([[[float(c[0][0] / shape[0]), float(c[0][1] / shape[1])]] for c in contour])
        # normalised_contour = []
        # for c in contour:
        #     x = float(c[0][0] / shape[1])
        #     y = float(c[0][1] / shape[0])
        #     normalised_contour.append([[x, y]])
        # normalised_contour = np.array(normalised_contour)
        return normalised_contour

    @property
    def ellipse_contour(self) -> np.ndarray | None:
        centre, axes, angle = cv2.fitEllipse(self.contour)
        centre = tuple(map(int, centre)) # Convert to integers
        axes = tuple(map(int, (axes[0] / 2, axes[1] / 2)))  # radius, not diameter
        ellipse_pts = cv2.ellipse2Poly(center=centre, axes=axes, angle=int(angle), arcStart=0, arcEnd=360, delta=5)
        ellipse_pts = np.array(ellipse_pts, dtype=np.float32)
        shape = self.entity.roi.size
        normalized_pts = np.array([[[float(x / shape[0]), float(y / shape[1])]] for x, y in ellipse_pts])
        return normalized_pts

    @staticmethod
    def bitmap2contour(out_mask: np.ndarray, contour_method: int = cv2.CHAIN_APPROX_SIMPLE) -> np.ndarray | None:
        """
        Returns the contour approximation of the mask using according to the specified method stored in self.contour_method
        """
        contour = None
        match contour_method:
            case cv2.CHAIN_APPROX_NONE:
                contour, _ = cv2.findContours(out_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            case cv2.CHAIN_APPROX_SIMPLE:
                contour, _ = cv2.findContours(out_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            case cv2.CHAIN_APPROX_TC89_L1:
                contour, _ = cv2.findContours(out_mask.astype(np.int32), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_TC89_L1)
            case cv2.CHAIN_APPROX_TC89_KCOS:
                contour, _ = cv2.findContours(out_mask.astype(np.int32), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_TC89_KCOS)
        if contour is not None:
            contour = max(contour, key=cv2.contourArea, default=None)
        return contour

    def change_mask(self, mask: np.ndarray) -> None:
        """
        changes the specified mask array
        :param mask: the new array for the mask
        """
        self.bin_mask = mask
        if mask is None:
            self.name = None
        self.project.save(self)

    def set_graphics_item(self, contour_method: int = cv2.CHAIN_APPROX_SIMPLE) -> None:
        """
        Sets the approximation method of contour from the binary mask accordingly and sets the brush
        :param contour_method: the contour approximation method to use
        """
        self.contour_method = contour_method
        self.graphics_item = self.to_shape()
        self.graphics_item.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setBrush()

    def to_shape(self) -> QGraphicsPolygonItem:
        """
        Returns a polygon approximation of the original binary mask
        """
        mask_shape = QPolygonF()
        for point in self.contour:
            mask_shape.append(QPointF(point[0][0], point[0][1]))
        return QGraphicsPolygonItem(mask_shape)

    @property
    def ellipse_item(self) -> QGraphicsEllipseItem:
        """
        return a graphics item to display the mask as an ellipse
        """
        if self._ellipse_item is None:
            e = cv2.fitEllipse(self.contour)
            ellipse_item = QGraphicsEllipseItem(e[0][0] - e[1][0] / 2.0, e[0][1] - e[1][1] / 2, e[1][0], e[1][1])
            ellipse_item.setTransformOriginPoint(e[0][0], e[0][1])
            ellipse_item.setRotation(e[2])
            self._ellipse_item = ellipse_item
            self._ellipse_item.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            return ellipse_item
        return self._ellipse_item

    @ellipse_item.setter
    def ellipse_item(self, item: QGraphicsEllipseItem) -> None:
        self._ellipse_item = item

    def setBrush(self, brush: QBrush = None) -> None:
        """
        Set the brush for all representation of the mask (polygon or ellipse)
        :param brush: the brush to use with this mask
        """
        if brush is not None:
            self.brush = brush
        self.graphics_item.setBrush(self.brush)
        self.ellipse_item.setBrush(self.brush)

    def record(self, no_id: bool = False) -> dict[str, Any]:
        """
        Returns a record dictionary of the current Entity

        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to another
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'uuid'    : self.uuid,
            'entity'  : self._entity,
            'name'    : self.name,
            'frame'   : self.frame,
            'bin_mask': self._bin_mask,
            'key_val' : self.key_val,
            }
        if not no_id:
            record['id_'] = self.id_
        return record
