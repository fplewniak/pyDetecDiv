"""
Image viewer to display and interact with an Image resource (5D image data)
"""

from PySide6.QtCore import Signal, Qt, QRect, QPoint, QTimer
from PySide6.QtGui import QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QFileDialog
import numpy as np
import cv2 as cv

from pydetecdiv.app.gui.core.widgets.viewers import Scene
from pydetecdiv.app.gui.core.widgets.viewers.images import ImageViewer
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer


class FOVmanager(VideoPlayer):
    """
    Class to view and manage a FOV
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.fov = None
        self.setup()

    def _create_viewer(self):
        viewer = ImageViewer()
        viewer.setup(FOVScene())
        return viewer

    def synchronize_with(self, other):
        """
        Synchronize the current viewer with another one. This is used to view a portion of an image in a new viewer
        with the same T coordinates as the original.
        :param other: the other viewer to synchronize with
        """
        if self.T != other.T:
            self.T = other.T
            self.video_frame.emit(self.T)
        self.viewer.display()

    def draw_saved_rois(self, roi_list):
        """
        Draw saved ROIs as green rectangles that can be selected but not moved

        :param roi_list: the list of saved ROIs
        :type roi_list: list of ROI objects
        """
        for roi in roi_list:
            rect_item = self.scene.addRect(QRect(0, 0, roi.width, roi.height))
            rect_item.setPen(self.scene.saved_pen)
            rect_item.setPos(QPoint(roi.x, roi.y))
            rect_item.setFlags(QGraphicsItem.ItemIsSelectable)
            rect_item.setData(0, roi.name)

    def load_roi_template(self):
        """
        Load ROI template from a file.
        """
        filename = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.tif *.tiff)")[0]
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        self.roi_template = np.uint8(np.array(img / np.max(img) * 255))
        self.ui.actionIdentify_ROIs.setEnabled(True)


class FOVScene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.match_pen = QPen(Qt.GlobalColor.yellow, 2)
        self.saved_pen = QPen(Qt.GlobalColor.green, 2)
        self.warning_pen = QPen(Qt.GlobalColor.red, 2)

    def duplicate_selected_Item(self, event):
        """
        Duplicate the currently selected Item at the current mouse position

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        item = super().duplicate_selected_Item(event)
        if item:
            self.check_is_colliding(item)
        return item

    def move_Item(self, event):
        """
        Move the currently selected Item if it is movable

        :param event: the mouse move event
        :type event: QGraphicsSceneMouseEvent
        """
        item = super().move_Item(event)
        if item and (item.flags() & QGraphicsItem.ItemIsMovable):
            self.check_is_colliding(item)
        return item

    def draw_Item(self, event):
        item = super().draw_Item(event)
        item.setPen(self.warning_pen)
        self.check_is_colliding(item)
        return item

    def check_is_colliding(self, item):
        if [r for r in item.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
            item.setPen(self.warning_pen)
        else:
            item.setPen(self.pen)
