"""
Image viewer to display and interact with an Image resource (5D image data)
"""

from PySide6.QtCore import Signal, Qt, QRect, QPoint, QTimer
from PySide6.QtGui import QPen, QAction, QTransform
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QFileDialog, QMenu, QMainWindow, QMenuBar
import numpy as np
import cv2 as cv
from skimage.feature import peak_local_max

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.core.widgets.viewers import Scene
from pydetecdiv.app.gui.core.widgets.viewers.images import ImageViewer
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer
from pydetecdiv.domain import Image, ROI


class FOVmanager(VideoPlayer):
    """
    Class to view and manage a FOV
    """

    def __init__(self, fov=None, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.fov = fov
        self.menuROI = None
        self.setup(menubar=self._create_menu_bar())

    def _create_menu_bar(self):
        menubar = QMenuBar()
        self.menuROI = menubar.addMenu('ROI')
        self.actionSet_template = QAction('Selection as template')
        self.menuROI.addAction(self.actionSet_template)
        self.actionSet_template.setEnabled(False)
        self.actionSet_template.triggered.connect(self.set_roi_template)

        self.actionLoad_template = QAction('Load template')
        self.menuROI.addAction(self.actionLoad_template)
        self.actionLoad_template.setEnabled(True)
        self.actionLoad_template.triggered.connect(self.load_roi_template)

        self.menuROI.addSeparator()

        self.actionIdentify_ROIs = QAction('Detect ROIs')
        self.menuROI.addAction(self.actionIdentify_ROIs)
        self.actionIdentify_ROIs.setEnabled(False)
        self.actionIdentify_ROIs.triggered.connect(self.identify_rois)

        self.actionSave_ROIs = QAction('Save ROIs')
        self.menuROI.addAction(self.actionSave_ROIs)
        self.actionSave_ROIs.setEnabled(False)
        self.actionSave_ROIs.triggered.connect(self.save_rois)
        return menubar

    def _create_viewer(self):
        viewer = ImageViewer()
        viewer.setup(FOVScene())
        return viewer

    def setImageResource(self, image_resource_data):
        # self.image_resource_data = image_resource_data
        self.setBackgroundImage(image_resource_data)

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

    def get_roi_image(self, roi):
        w, h = roi.rect().toRect().size().toTuple()
        pos = roi.pos()
        x1, x2 = int(pos.x()), w + int(pos.x())
        y1, y2 = int(pos.y()), h + int(pos.y())
        (C, T, Z) = self.viewer.background.image.get_CTZ()
        return Image.auto_channels(self.viewer.image_resource_data, C=C, T=T, Z=Z, crop=(slice(x1, x2), slice(y1, y2)),
                                   drift=PyDetecDiv.apply_drift, alpha=False).as_array(np.uint8)

    def set_roi_template(self):
        """
        Set the currently selected area as a template to define other ROIs
        """
        roi = self.scene.get_selected_Item()
        if roi:
            PyDetecDiv.roi_template = self.get_roi_image(roi)
            self.actionIdentify_ROIs.setEnabled(True)

    def load_roi_template(self):
        """
        Load ROI template from a file.
        """
        filename = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.tif *.tiff)")[0]
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        PyDetecDiv.roi_template = np.uint8(np.array(img / np.max(img) * 255))
        self.actionIdentify_ROIs.setEnabled(True)

    def identify_rois(self):
        """
        Identify ROIs in an image using the ROI template as a model and the matchTemplate function from OpenCV
        """
        threshold = 0.3
        (C, T, Z) = self.viewer.background.image.get_CTZ()
        img = Image.auto_channels(self.viewer.image_resource_data, C=C, Z=Z, T=T, alpha=False).as_array(np.uint8)
        res = cv.matchTemplate(img, PyDetecDiv.roi_template, cv.TM_CCOEFF_NORMED)
        xy = peak_local_max(res, threshold_abs=threshold, exclude_border=False)
        w, h = PyDetecDiv.roi_template.shape[::-1]
        for pt in xy:
            x, y = pt[1], pt[0]
            if not isinstance(self.scene.itemAt(QPoint(x, y), QTransform().scale(1, 1)), QGraphicsRectItem):
                rect_item = self.scene.addRect(QRect(0, 0, w, h))
                rect_item.setPos(x, y)
                if [r for r in rect_item.collidingItems(Qt.IntersectsItemBoundingRect) if
                    isinstance(r, QGraphicsRectItem)]:
                    self.scene.removeItem(rect_item)
                else:
                    rect_item.setPen(self.scene.match_pen)
                    rect_item.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.actionSave_ROIs.setEnabled(True)

    def save_rois(self):
        """
        Save the areas as ROIs
        """
        print(self.fov)
        rois = [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem)]
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            roi_list = [r.name for r in self.fov.roi_list]
            for i, rect_item in enumerate(sorted(rois, key=lambda x: x.scenePos().toPoint().toTuple())):
                x, y = rect_item.scenePos().toPoint().toTuple()
                w, h = rect_item.rect().toRect().getCoords()[2:]
                new_roi_name = f'{self.fov.name}_{x}_{y}_{w}_{h}'
                if new_roi_name not in roi_list:
                    new_roi = ROI(project=project, name=new_roi_name, fov=self.fov,
                                  top_left=(x, y), bottom_right=(int(x) + w, int(y) + h))
                    rect_item.setData(0, new_roi.name)
        PyDetecDiv.app.saved_rois.emit(PyDetecDiv.project_name)
        self.fixate_saved_rois()
        self.actionSave_ROIs.setEnabled(False)

    def fixate_saved_rois(self):
        """
        Disable the possibility to move ROIs once they have been saved
        """
        for r in [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem)]:
            r.setPen(self.scene.saved_pen)
            r.setFlag(QGraphicsItem.ItemIsMovable, False)


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
        if self.get_colliding_ShapeItems(item):
            item.setPen(self.warning_pen)
        else:
            item.setPen(self.pen)
