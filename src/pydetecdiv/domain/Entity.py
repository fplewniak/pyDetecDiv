#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Entities
"""
import sys
from typing import Any

from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.dso import NamedDSO


class Entity(NamedDSO):
    """
    A class defining an entity that can be segmented, tracked, etc in a time-lapse microscopy video
    """

    def __init__(self, roi: int | ROI, category: str = 'cell', exit_frame: int = sys.maxsize, key_val: dict = None, **kwargs):
        super().__init__(**kwargs)
        self._roi = roi.id_ if isinstance(roi, ROI) else roi
        self._exit_frame = exit_frame
        self.category = category
        self.key_val = key_val
        self.validate(updated=False)

    @property
    def roi(self) -> ROI:
        """
        property returning the ROI object this entity belongs to

        :return: the parent ROI object
        """
        return self.project.get_object('ROI', self._roi)

    @roi.setter
    def roi(self, roi: int | ROI) -> None:
        self._roi = roi.id_ if isinstance(roi, ROI) else roi
        self.validate()

    @property
    def exit_frame(self) -> int:
        return self._exit_frame

    @exit_frame.setter
    def exit_frame(self, exit_frame: int) -> None:
        self._exit_frame = exit_frame
        self.validate()

    def bounding_box(self, frame: int):
        if self.bounding_boxes(frame=frame):
            return self.bounding_boxes(frame=frame)[0]
        return None

    def bounding_boxes(self, frame: int = None):
        all_bounding_boxes = self.project.get_linked_objects('BoundingBox', self)
        return [bb for bb in all_bounding_boxes if bb.frame == frame] if frame is not None else all_bounding_boxes

    def points(self, frame: int = None):
        all_points = self.project.get_linked_objects('Point', self)
        return [p for p in all_points if p.frame == frame] if frame is not None else all_points

    def masks(self, frame: int = None):
        all_masks = self.project.get_linked_objects('Mask', self)
        return [m for m in all_masks if m.frame == frame] if frame is not None else all_masks

    def mask(self, frame: int):
        if self.masks(frame=frame):
            return self.masks(frame=frame)[0]
        return None

    def record(self, no_id: bool = False) -> dict[str, Any]:
        """
        Returns a record dictionary of the current Entity

        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to another
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'name'      : self.name,
            'roi'       : self._roi,
            'uuid'      : self.uuid,
            'category'  : self.category,
            'exit_frame': self._exit_frame,
            'key_val'   : self.key_val,
            }
        if not no_id:
            record['id_'] = self.id_
        return record
