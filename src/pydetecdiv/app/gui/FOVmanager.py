"""
Image viewer to display and interact with an Image resource (5D image data)
"""

from PySide6.QtCore import Signal, Qt, QRect, QPoint
from PySide6.QtGui import QPen, QAction, QTransform
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QFileDialog, QMenu, QMenuBar, QWidget, QGraphicsSceneMouseEvent
import numpy as np
import cv2 as cv
from skimage.feature import peak_local_max

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.core.widgets.viewers import Scene
from pydetecdiv.app.gui.core.widgets.viewers.annotation.sam2.segmentation_tool import SegmentationTool, SegmentationScene
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer
from pydetecdiv.domain.Image import Image, ImgDType
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.ImageResourceData import ImageResourceData


class FOVmanager(VideoPlayer):
    """
    Class to view and manage a FOV
    """

    def __init__(self, fov=None, parent: QWidget = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.fov = fov
        self.menuROI = None
        self.setup(menubar=self.create_menubar())

    # def other_scene_in_focus(self, tab):
    #     if PyDetecDiv.main_window.active_subwindow.widget(tab).scene == self.scene:
    #         PyDetecDiv.app.other_scene_in_focus.emit(self.scene)

    def create_menubar(self) -> QMenuBar | None:
        """
        Adds a menu bar to the current widget

        :return: the menubar
        """
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

    # def _create_viewer(self):
    #     """
    #     Creates a viewer with a FOVScene instead of a Scene
    #
    #     :return: the created viewer
    #     """
    #     viewer = ImageViewer()
    #     viewer.setup(FOVScene())
    #     viewer.scene().roi_selected.connect(self.actionSet_template.setEnabled)
    #     # viewer.scene().roi_selected.connect(self.actionIdentify_ROIs.setEnabled)
    #     viewer.scene().not_saved_rois.connect(self.actionSave_ROIs.setEnabled)
    #     return viewer

    def setup(self, menubar: QMenuBar = None) -> None:
        super().setup(menubar=menubar)
        self.viewer_panel.setup(scene=FOVScene())
        self.viewer.scene().roi_selected.connect(self.actionSet_template.setEnabled)
        self.viewer.scene().not_saved_rois.connect(self.actionSave_ROIs.setEnabled)

    def setImageResource(self, image_resource_data: ImageResourceData, C: int = 0, Z: int = 0, T: int = 0) -> None:
        """
        Sets the Image Resource

        :param image_resource_data: the Image resource data
        :param C: the channel or channel tuple to display
        :param Z: the Z-slice (or tuple to combine z-slices as channels)
        :param T: the initial time frame
        """
        # self.image_resource_data = image_resource_data
        self.setBackgroundImage(image_resource_data, C=C, Z=Z, T=T)

    def draw_saved_rois(self, roi_list: list[ROI]) -> None:
        """
        Draw saved ROIs as green rectangles that can be selected but not moved

        :param roi_list: the list of saved ROIs
        :type roi_list: list of ROI objects
        """
        for roi in roi_list:
            rect_item = self.scene.addRect(QRect(0, 0, roi.width, roi.height))
            rect_item.setPen(self.scene.saved_pen)
            rect_item.setPos(QPoint(roi.x, roi.y))
            rect_item.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            rect_item.setData(0, roi.name)
            rect_item.setZValue(len(self.viewer.layers))
        PyDetecDiv.app.other_scene_in_focus.emit(self.scene)

    def get_roi_image(self, roi: QGraphicsRectItem) -> np.ndarray:
        """
        Get an Image of the defined ROI

        :param roi: the ROI
        :return: Image of the ROI
        """
        w, h = roi.rect().toRect().size().toTuple()
        pos = roi.pos()
        x1, x2 = int(pos.x()), w + int(pos.x())
        y1, y2 = int(pos.y()), h + int(pos.y())
        (C, T, Z) = self.viewer.background.image.get_CTZ()
        return Image.auto_channels(self.viewer.image_resource_data, C=C, T=T, Z=Z, crop=(slice(x1, x2), slice(y1, y2)),
                                   drift=PyDetecDiv.apply_drift, alpha=False).as_array(ImgDType.uint8)

    def view_in_new_tab(self, rect: QGraphicsRectItem) -> None:
        """
        view a selection in a new tab
        :param rect: the rectangular selection to view
        """
        self._view_in_new_tab(rect, VideoPlayer(), scene=Scene())

    def _view_in_new_tab(self, rect: QGraphicsRectItem, player: VideoPlayer, scene: Scene = None) -> None:
        """
        view a selection in a new tab with a given player and scene
        """
        video_player = PyDetecDiv.main_window.active_subwindow.widget(
                PyDetecDiv.main_window.active_subwindow.addTab(player, rect.data(0)))
        scene.player = video_player
        video_player.tscale = self.tscale
        video_player.setup(menubar=video_player.create_menubar())
        video_player.viewer_panel.setup(scene=scene)
        w, h = rect.rect().toRect().size().toTuple()
        pos = rect.pos()
        x1, x2 = int(pos.x()), w + int(pos.x())
        y1, y2 = int(pos.y()), h + int(pos.y())
        (C, T, Z) = self.viewer.background.image.get_CTZ()
        video_player.setBackgroundImage(self.viewer.image_resource_data, C=C, Z=Z, T=T,
                                        crop=(slice(x1, x2), slice(y1, y2)))
        for layer in self.viewer.layers[1:]:
            (C, T, Z) = layer.image.get_CTZ()
            video_player.addLayer().setImage(self.viewer.image_resource_data, C=C, Z=Z, T=T,
                                             crop=(slice(x1, x2), slice(y1, y2)), alpha=True)
        PyDetecDiv.main_window.active_subwindow.setCurrentWidget(video_player)
        PyDetecDiv.app.other_scene_in_focus.emit(scene)
        video_player.reset()

    def open_in_segmentation_tool(self, rect: QGraphicsRectItem) -> None:
        """
        open the selected rectangle area (which should actually be a ROI) in the segmentation tool
        """
        self._view_in_new_tab(rect, SegmentationTool(rect.data(0)), scene=SegmentationScene())
        # PyDetecDiv.main_window.scene_tree_palette.tree_view.setModel(SegmentationTreeModel(['boxes']))
        # PyDetecDiv.main_window.scene_tree_palette.set_top_items([])
        # PyDetecDiv.main_window.active_subwindow.currentWidget().create_video(rect.data(0))

    def set_roi_template(self) -> None:
        """
        Set the currently selected area as a template to define other ROIs
        """
        roi = self.scene.get_selected_Item()
        if roi:
            PyDetecDiv.roi_template = self.get_roi_image(roi)
            self.actionIdentify_ROIs.setEnabled(True)

    def load_roi_template(self) -> None:
        """
        Load ROI template from a file.
        """
        filename = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.tif *.tiff)")[0]
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        PyDetecDiv.roi_template = np.uint8(np.array(img / np.max(img) * 255))
        self.actionIdentify_ROIs.setEnabled(True)

    def identify_rois(self) -> None:
        """
        Identify ROIs in an image using the ROI template as a model and the matchTemplate function from OpenCV
        """
        threshold = 0.3
        (C, T, Z) = self.viewer.background.image.get_CTZ()
        img = Image.auto_channels(self.viewer.image_resource_data, C=C, Z=Z, T=T, alpha=False).as_array(ImgDType.uint8)
        res = cv.matchTemplate(img, PyDetecDiv.roi_template, cv.TM_CCOEFF_NORMED)
        xy = peak_local_max(res, threshold_abs=threshold, exclude_border=False)
        w, h = PyDetecDiv.roi_template.shape[::-1]
        for pt in xy:
            x, y = pt[1], pt[0]
            if not isinstance(self.scene.itemAt(QPoint(x, y), QTransform().scale(1, 1)), QGraphicsRectItem):
                rect_item = self.scene.addRect(QRect(0, 0, w, h))
                rect_item.setPos(x, y)
                rect_item.setData(0, f'{self.fov.name}_{x}_{y}_{w + 1}_{h + 1}')
                if [r for r in rect_item.collidingItems(Qt.ItemSelectionMode.IntersectsItemBoundingRect) if
                    isinstance(r, QGraphicsRectItem)]:
                    self.scene.removeItem(rect_item)
                else:
                    rect_item.setPen(self.scene.match_pen)
                    rect_item.setFlags(
                            QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        PyDetecDiv.app.scene_modified.emit(self.viewer.scene())
        self.actionSave_ROIs.setEnabled(True)

    def save_rois(self) -> None:
        """
        Save the areas as ROIs
        """
        rois = [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem) and item.pen() != self.scene.saved_pen]
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            roi_list = [r.name for r in self.fov.roi_list]
            for rect_item in sorted(rois, key=lambda x: x.scenePos().toPoint().toTuple()):
                x, y = rect_item.scenePos().toPoint().toTuple()
                w, h = rect_item.rect().toRect().getCoords()[2:]
                new_roi_name = f'{self.fov.name}_{x}_{y}_{w + 1}_{h + 1}'
                if new_roi_name not in roi_list:
                    new_roi = ROI(project=project, name=new_roi_name, fov=self.fov,
                                  top_left=(x, y), bottom_right=(int(x) + w, int(y) + h))
                    rect_item.setData(0, new_roi.name)
        PyDetecDiv.app.saved_rois.emit(PyDetecDiv.project_name)
        PyDetecDiv.app.scene_modified.emit(self.scene)
        PyDetecDiv.app.other_scene_in_focus.emit(self.scene)
        self.fixate_saved_rois()
        self.actionSave_ROIs.setEnabled(False)

    def fixate_saved_rois(self) -> None:
        """
        Disable the possibility to move ROIs once they have been saved
        """
        for r in [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem)]:
            r.setPen(self.scene.saved_pen)
            r.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)


class FOVScene(Scene):
    """
    A class describing a Scene displaying a FOV
    """
    roi_selected = Signal(bool)
    not_saved_rois = Signal(bool)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.match_pen = QPen(Qt.GlobalColor.yellow, 2)
        self.saved_pen = QPen(Qt.GlobalColor.green, 2)
        self.warning_pen = QPen(Qt.GlobalColor.red, 2)

    def removeItem(self, item: QGraphicsItem) -> None:
        """
        Remove the item from the Scene

        :param item: the item to remove
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            roi = project.get_named_object("ROI", item.data(0))
            if roi is not None:
                roi.delete()
        super().removeItem(item)
        self.not_saved_rois.emit(self.check_not_saved_rois())

    def select_Item(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        Select the current area/Item

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        super().select_Item(event)
        self.not_saved_rois.emit(self.check_not_saved_rois())
        self.roi_selected.emit(self.get_selected_Item() is not None)

    def duplicate_selected_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem:
        """
        Duplicate the currently selected Item at the current mouse position

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        item = super().duplicate_selected_Item(event)
        self.select_Item(event)
        if item:
            self.check_is_colliding(item)
            self.roi_selected.emit(self.get_selected_Item() is not None)
        self.not_saved_rois.emit(self.check_not_saved_rois())
        return item

    def move_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem:
        """
        Move the currently selected Item if it is movable

        :param event: the mouse move event
        :type event: QGraphicsSceneMouseEvent
        """
        item = super().move_Item(event)
        if item and (item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable):
            self.check_is_colliding(item)
            self.roi_selected.emit(self.get_selected_Item() is not None)
        return item

    def draw_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem:
        """
        Draws an Item

        :param event: the mouse move event
        :type event: QGraphicsSceneMouseEvent
        """
        item = super().draw_Item(event)
        if item:
            item.setPen(self.warning_pen)
            self.check_is_colliding(item)
            self.roi_selected.emit(self.get_selected_Item() is not None)
            self.not_saved_rois.emit(self.check_not_saved_rois())
        return item

    def contextMenuEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        The context menu for area manipulation

        :param event:
        """
        menu = QMenu()
        view_in_new_tab = menu.addAction("View in new tab")
        open_in_segmentation_tool = menu.addAction("Manual segmentation")
        open_in_segmentation_tool.setEnabled(False)
        roi = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            if project.get_named_object('ROI', roi.data(0)):
                open_in_segmentation_tool.setEnabled(True)
        if isinstance(roi, QGraphicsRectItem):
            # view_in_new_tab.triggered.connect(lambda _: self.parent().view_roi_image(r))
            view_in_new_tab.triggered.connect(
                    lambda _: PyDetecDiv.main_window.active_subwindow.currentWidget().view_in_new_tab(roi))
            open_in_segmentation_tool.triggered.connect(
                    lambda _: PyDetecDiv.main_window.active_subwindow.currentWidget().open_in_segmentation_tool(roi))
            PyDetecDiv.app.viewer_roi_click.emit((roi, menu))
            menu.exec(event.screenPos())

    def check_is_colliding(self, item: QGraphicsRectItem) -> None:
        """
        Checks whether the item collides with another one. If it does, set its pen to warning pen

        :param item: the item to check
        """
        if isinstance(item, QGraphicsRectItem) and [i for i in self.get_colliding_ShapeItems(item) if
                                                    isinstance(i, QGraphicsRectItem)]:
            item.setPen(self.warning_pen)
        else:
            item.setPen(self.pen)

    def check_not_saved_rois(self) -> bool:
        """
        Return True if there is at least one unsaved ROI

        :return: bool
        """
        return any(item.pen() in [self.pen, self.match_pen] for item in self.items() if isinstance(item, QGraphicsRectItem))
