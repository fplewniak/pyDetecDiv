"""
Computer Assisted Segmentation Tool: a tool for manual segmentation of images (annotation) using SegmentAnything2 by META to
segment and propagate segmentation using prompts (bounding boxes, points and masks)
"""
import gc
import os

import cv2
from PySide6.QtCore import Qt, QModelIndex, QPointF
from PySide6.QtGui import (QPen, QKeyEvent, QKeySequence, QStandardItem, QCloseEvent, QPolygonF, QBrush, QColor, QGuiApplication,
                           QTransform, QActionGroup, QAction)
from PySide6.QtWidgets import (QGraphicsSceneMouseEvent, QMenu, QWidget, QGraphicsEllipseItem, QMenuBar, QVBoxLayout, QLabel,
                               QHBoxLayout, QSplitter, QGraphicsRectItem, QGraphicsItem, QGraphicsPolygonItem, QSizePolicy)
import numpy as np
import torch.cuda
from sam2.build_sam import build_sam2_video_predictor
from pydetecdiv.app import PyDetecDiv, DrawingTools
from pydetecdiv.app.gui.core.widgets.viewers.annotation.sam2.objectsmodel import (ObjectsTreeView, Object, PromptProxyModel,
                                                                                  PromptSourceModel, ObjectReferenceRole,
                                                                                  BoundingBox, Point, ModelItem, Mask)
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer, VideoViewerPanel, VideoControlPanel, VideoScene
from pydetecdiv.settings import get_config_value


class Colours:
    """
    Colour definition for brushes used to display Mask graphics items
    """
    palette = [
        QColor(255, 0, 0, 100),
        QColor(0, 255, 0, 100),
        QColor(0, 0, 255, 100),
        QColor(128, 0, 0, 100),
        QColor(0, 128, 0, 100),
        QColor(0, 0, 128, 100),
        QColor(128, 0, 128, 100),
        QColor(128, 128, 0, 100),
        QColor(0, 128, 128, 100),
        QColor(128, 0, 255, 100),
        QColor(255, 128, 0, 100),
        QColor(0, 255, 128, 100),
        QColor(255, 0, 128, 100),
        QColor(128, 255, 0, 100),
        QColor(0, 128, 255, 100),
        QColor(64, 0, 0, 100),
        QColor(0, 64, 0, 100),
        QColor(0, 0, 64, 100),
        QColor(64, 0, 255, 100),
        QColor(64, 255, 0, 100),
        QColor(0, 64, 255, 100),
        QColor(255, 64, 0, 100),
        QColor(0, 255, 64, 100),
        QColor(255, 0, 64, 100),
        QColor(64, 0, 128, 100),
        QColor(64, 128, 0, 100),
        QColor(0, 64, 128, 100),
        QColor(128, 64, 0, 100),
        QColor(0, 128, 64, 100),
        QColor(128, 0, 64, 100),
        ]


class SegmentationScene(VideoScene):
    """
    A class handling the VideoScene for the Segmentation tool
    """

    def __init__(self, parent: QWidget = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.default_pen = QPen(Qt.GlobalColor.green, 1)
        self.positive_pen = QPen(Qt.GlobalColor.green, 1)
        self.negative_pen = QPen(Qt.GlobalColor.red, 1)
        self.pen = self.default_pen

        self.object_list = []
        self.last_shape = None

    @property
    def current_object(self) -> Object:
        """
        The currently selected object. All prompt item additions will be attached to this object.
        """
        return self.player.current_object

    def reset_graphics_items(self) -> None:
        """
        Clear the scene before drawing new graphics item. This is used when changing frame, to ensure only items defined for the
        current frame are displayed
        """
        for item in self.items():
            if isinstance(item, (QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPolygonItem)):
                self.removeItem(item)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Detect when a key is pressed and perform the corresponding action:
        * QKeySequence.Delete: delete the selected item

        :param event: the key pressed event
        :type event: QKeyEvent
        """
        if event.matches(QKeySequence.StandardKey.Delete):
            for r in self.selectedItems():
                self.delete_item(r)
        elif event.key() == Qt.Key.Key_Insert:
            current_object = Object(len(self.object_list))
            self.object_list.append(current_object)
            self.player.add_object(current_object)
            # self.player.prompt_model.add_object(current_object)

    def select_from_tree_view(self, graphics_item: QGraphicsItem) -> None:
        """
        Selection of a graphics item responding to a click in the tree view
        :param graphics_item: the graphics item to select
        """
        _ = [r.setSelected(False) for r in self.selectedItems()]
        if graphics_item is not None:
            graphics_item.setSelected(True)
            if isinstance(graphics_item, (QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPolygonItem)):
                self.player.object_tree_view.select_object_from_graphics_item(graphics_item, self.player.T)

    def select_Item(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        method overriding the select_Item method of QGraphicsScene in order to synchronize the selection in the Scene with a
        selection in the tree view.
        :param event:
        """
        graphics_item = super().select_Item(event)
        if isinstance(graphics_item, (QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPolygonItem)):
            self.player.object_tree_view.select_object_from_graphics_item(graphics_item, self.player.T)

    def delete_item(self, graphics_item: QGraphicsItem) -> None:
        """
        Delete a graphics item
        :param graphics_item: the graphics item to delete
        """
        if isinstance(graphics_item, QGraphicsRectItem):
            self.player.source_model.remove_bounding_box(self.current_object, self.player.T)
        if isinstance(graphics_item, QGraphicsEllipseItem):
            self.player.source_model.remove_point(self.current_object, graphics_item, self.player.T)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        Reactions to a mouse click release event in the scene, actualizing bounding box creation or modification
        :param event: the QGraphicsSceneMouseEvent
        """
        if event.button() == Qt.MouseButton.LeftButton:
            match PyDetecDiv.current_drawing_tool, event.modifiers():
                case DrawingTools.DrawRect, Qt.KeyboardModifier.NoModifier:
                    if self.last_shape:
                        if self.player.source_model.has_bounding_box(self.current_object, self.player.T):
                            # self.update_Item_size(self.last_shape)
                            self.player.source_model.change_bounding_box(self.current_object, self.player.T, self.last_shape)
                        else:
                            self.player.source_model.add_bounding_box(self.current_object, self.player.T, self.last_shape)
                        self.player.object_tree_view.expandAll()
                        self.last_shape = None
                case DrawingTools.DuplicateItem, Qt.KeyboardModifier.NoModifier:
                    if self.selectedItems():
                        rect_item = self.duplicate_selected_Item(event)
                        self.player.source_model.change_bounding_box(self.current_object, self.player.T, rect_item)
                        self.player.object_tree_view.expandAll()
                        self.select_Item(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        Detect when the left mouse button is pressed and perform the action corresponding to the currently checked
        drawing tool

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        if event.button() == Qt.MouseButton.LeftButton:
            match PyDetecDiv.current_drawing_tool, event.modifiers():
                case DrawingTools.Cursor, Qt.KeyboardModifier.NoModifier:
                    self.select_Item(event)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.NoModifier:
                    self.unselect_items(event)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.ControlModifier:
                    self.select_Item(event)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.ShiftModifier:
                    self.select_Item(event)
                # case DrawingTools.DuplicateItem, Qt.KeyboardModifier.NoModifier:
                #     self.duplicate_selected_Item(event)
                case DrawingTools.DrawPoint, Qt.KeyboardModifier.NoModifier:
                    self.add_point(event)
                    self.player.object_tree_view.expandAll()
                case DrawingTools.DrawPoint, Qt.KeyboardModifier.ControlModifier:
                    self.add_point(event)
                    self.player.object_tree_view.expandAll()
                case DrawingTools.DrawPoint, Qt.KeyboardModifier.ShiftModifier:
                    self.select_Item(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        Detect mouse movement and apply the appropriate method according to the currently checked drawing tool and key
        modifier

        :param event: the mouse move event
        :type event: QGraphicsSceneMouseEvent
        """
        if event.buttons() == Qt.MouseButton.LeftButton:
            match PyDetecDiv.current_drawing_tool, event.modifiers():
                case DrawingTools.Cursor, Qt.KeyboardModifier.NoModifier:
                    self.move_Item(event)
                case DrawingTools.Cursor, Qt.KeyboardModifier.ControlModifier:
                    self.draw_Item(event)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.NoModifier:
                    self.draw_Item(event)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.ControlModifier:
                    self.move_Item(event)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.ShiftModifier:
                    self.draw_Item(event)
                case DrawingTools.DuplicateItem, Qt.KeyboardModifier.NoModifier:
                    self.move_Item(event)

    def move_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem | QGraphicsEllipseItem:
        """
        Move the selected graphics item
        :param event: the QGraphicsSceneMouseEvent
        :return: the moved graphics item
        """
        graphics_item = super().move_Item(event)
        self.update_Item_coordinates(graphics_item)
        return graphics_item

    def update_Item_coordinates(self, graphics_item: QGraphicsRectItem | QGraphicsEllipseItem) -> None:
        """
        Update the coordinates of a graphics item in the source model for display in the tree view, after this item has been moved
        :param graphics_item: the moved graphics item
        """
        x_item = self.player.source_model.graphics2model_item(graphics_item, self.player.T, 2)
        y_item = self.player.source_model.graphics2model_item(graphics_item, self.player.T, 3)
        if x_item is not None:
            x_item.setData(f'{graphics_item.pos().x():.1f}', 0)
        if y_item is not None:
            y_item.setData(f'{graphics_item.pos().y():.1f}', 0)

    def update_Item_size(self, graphics_item: QGraphicsRectItem) -> None:
        """
        Update the bounding box size information in the source model for display in the tree view after the box has been resized
        :param graphics_item: the bounding box graphics item
        """
        width_item = self.player.source_model.graphics2model_item(graphics_item, self.player.T, 4)
        height_item = self.player.source_model.graphics2model_item(graphics_item, self.player.T, 5)
        if width_item is not None:
            width_item.setData(f'{int(graphics_item.rect().width())}', 0)
        if height_item is not None:
            height_item.setData(f'{int(graphics_item.rect().height())}', 0)

    def draw_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem:
        """
        Draw a rectangular item representing a bounding box
        :param event: the mouse event
        :return: the rectangular item
        """
        if self.player.current_object is not None:
            self.last_shape = super().draw_Item(event)
            self.last_shape.setData(0, f'bounding_box{self.player.current_object.id_}')
            self.update_Item_size(self.last_shape)
        else:
            self.last_shape = None
        return self.last_shape

    def duplicate_selected_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem | None:
        """
        Duplicate the selected bounding box
        :param event: the mouse event
        :return: the duplicated item
        """
        if self.player.current_object is not None:
            item = super().duplicate_selected_Item(event)
            item.setData(0, f'bounding_box{self.player.current_object.id_}')
            return item
        return None

    def contextMenuEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        The context menu

        :param event: mouse event
        """
        menu = QMenu()
        # frame_segment_action = menu.addAction('Run segmentation on current frame')
        # frame_segment_action.triggered.connect(self.player.segment_from_prompt)
        video_segment_action = menu.addAction('Run segmentation on video')
        video_segment_action.triggered.connect(self.player.segment_from_prompt)
        item_at_click = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
        if isinstance(item_at_click, (QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPolygonItem)):
            self.select_Item(event)
            exit_next_frame_action = menu.addAction('Set exit frame as next frame')
            exit_next_frame_action.triggered.connect(self.player.object_exit_next_frame)
        menu.exec(event.screenPos())

    def add_point(self, event: QGraphicsSceneMouseEvent) -> QGraphicsEllipseItem | None:
        """
        Add a point (positive is green, negative is red) defining the interior of an object to segment
        :param event: the mouse event
        :return: the point (small ellipse) graphics item
        """
        if self.current_object is not None:
            label = 1
            self.pen = self.positive_pen
            if event.buttons() == Qt.MouseButton.LeftButton:
                match PyDetecDiv.current_drawing_tool, event.modifiers():
                    case DrawingTools.DrawPoint, Qt.KeyboardModifier.ControlModifier:
                        self.pen = self.negative_pen
                        label = 0
            item = super().add_point(event)
            item.setData(1, label)
            self.pen = self.default_pen
            self.player.source_model.add_point(self.current_object, self.player.T, item, label)
            return item
        return None


class SegmentationTool(VideoPlayer):
    """
    Annotator class extending the VideoPlayer class to define functionalities specific to ROI image annotation
    """

    def __init__(self, region_name: str):
        super().__init__()
        self.region = region_name
        self.run = None
        self.viewport_rect = None
        self.source_model = PromptSourceModel()
        self.proxy_model = PromptProxyModel()
        self.proxy_model.setRecursiveFilteringEnabled(True)
        self.predictor = None
        self.inference_state = None
        self.video_segments = None
        self.out_mask = None
        self.mask_logits = None
        self.object_tree_view = None
        self.max_frames_prop = 15
        self.method_group = None
        self.no_approximation = None
        self.simple_approximation = None
        self.TCL1_approximation = None
        self.TCKCOS_approximation = None
        self.display_ellipses = None

    @property
    def contour_method(self) -> int:
        """
        Returns the selected contour approximation method to use to convert the binary masks to contours
        """
        checked_action = self.method_group.checkedAction()
        match checked_action:
            case self.no_approximation:
                return cv2.CHAIN_APPROX_NONE
            case self.simple_approximation:
                return cv2.CHAIN_APPROX_SIMPLE
            case self.TCL1_approximation:
                return cv2.CHAIN_APPROX_TC89_L1
            case self.TCKCOS_approximation:
                return cv2.CHAIN_APPROX_TC89_KCOS

    @property
    def current_tree_index(self) -> QModelIndex:
        """
        the source index of the current selection obtained from the proxy model
        :return:
        """
        return self.proxy_model.mapToSource(self.object_tree_view.currentIndex())

    @property
    def current_tree_item(self) -> QStandardItem:
        """
        the currently selected item
        :return:
        """
        return self.source_model.itemFromIndex(self.current_tree_index)

    @property
    def current_object(self) -> Object | None:
        """
        Returns the current object according to the selection in the tree view
        :return: the current object
        """
        item = self.current_tree_item
        if item is not None:
            obj = item.data(ObjectReferenceRole)
            if isinstance(obj, Object):
                return obj
            if isinstance(obj, (BoundingBox, Point)):
                return item.parent().data(ObjectReferenceRole)
        return None

    @property
    def prompt(self) -> dict:
        """
        Returns the prompt for all objects in the current frame
        :return: the prompt
        """
        return {obj.id_: {self.T: self.source_model.get_prompt_for_key_frame(obj, self.T)}
                for obj in self.source_model.objects if self.T in self.source_model.key_frames(obj)}

    def add_object(self, obj: Object) -> ModelItem:
        """
        Add a new object
        :param obj: the new object
        :return: the model item corresponding to the added object
        """
        new_item = self.source_model.add_object(obj)
        self.object_tree_view.select_item(new_item)
        self.proxy_model.invalidateFilter()
        return new_item

    def create_menubar(self) -> QMenuBar | None:
        menubar = QMenuBar()
        menubar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        maskApproximation = menubar.addMenu('Mask approximation')
        self.method_group = QActionGroup(maskApproximation)
        self.no_approximation = QAction('No approximation')
        self.no_approximation.setActionGroup(self.method_group)
        self.no_approximation.setCheckable(True)
        self.simple_approximation = QAction('Simple approximation')
        self.simple_approximation.setActionGroup(self.method_group)
        self.simple_approximation.setCheckable(True)
        self.TCL1_approximation = QAction('Teh Chin L1  approximation')
        self.TCL1_approximation.setActionGroup(self.method_group)
        self.TCL1_approximation.setCheckable(True)
        self.TCKCOS_approximation = QAction('Teh Chin KCOS approximation')
        self.TCKCOS_approximation.setActionGroup(self.method_group)
        self.TCKCOS_approximation.setCheckable(True)
        maskApproximation.addAction(self.no_approximation)
        maskApproximation.addAction(self.simple_approximation)
        maskApproximation.addAction(self.TCL1_approximation)
        maskApproximation.addAction(self.TCKCOS_approximation)
        maskApproximation.addSeparator()
        self.display_ellipses = QAction('Fit ellipse')
        self.display_ellipses.setCheckable(True)
        maskApproximation.addAction(self.display_ellipses)
        self.simple_approximation.setChecked(True)

        self.method_group.triggered.connect(self.redraw_scene)
        self.display_ellipses.toggled.connect(self.redraw_scene)
        return menubar

    def setup(self, menubar: QMenuBar = None) -> None:
        """
        Sets the video player up

        :param menubar: whether a menubar should be added or not
        """
        layout = QVBoxLayout(self)
        if menubar:
            self.menubar = menubar
            layout.addWidget(self.menubar)

        video_widget = QWidget(self)
        video_layout = QVBoxLayout(video_widget)
        self.viewer_panel = VideoViewerPanel(self)
        self.control_panel = VideoControlPanel(self)
        video_layout.addWidget(self.viewer_panel)
        video_layout.addWidget(self.control_panel)
        video_widget.setLayout(video_layout)

        self.object_tree_view = ObjectsTreeView()
        self.object_tree_view.setModel(self.proxy_model)
        self.object_tree_view.setSourceModel(self.source_model)
        self.object_tree_view.setup()
        self.object_tree_view.pressed.connect(self.select_from_tree_view)
        # self.object_tree_view.object_deleted.connect(self.delete_object)

        splitter = QSplitter()
        splitter.addWidget(video_widget)
        splitter.addWidget(self.object_tree_view)

        # Set the initial stretch factors
        splitter.setStretchFactor(0, 2)  # Left widget is twice as wide
        splitter.setStretchFactor(1, 1)  # Right widget is normal width

        h_layout = QHBoxLayout()
        h_layout.addWidget(splitter)

        layout.addLayout(h_layout)
        self.setLayout(layout)

        self.time_display = QLabel(self.elapsed_time, parent=self)
        self.time_display.setStyleSheet("color: green; font-size: 18px;")
        self.time_display.setGeometry(20, 40, 140, self.time_display.height())

        self.video_frame.connect(self.proxy_model.set_frame)

    def object_exit_next_frame(self) -> None:
        """
        Sets the exit frame of the current object to the next frame.
        """
        if self.T < self.viewer.image_resource_data.sizeT - 1:
            self.source_model.object_exit(self.current_object, self.T + 1)

    def select_from_tree_view(self, index: QModelIndex) -> None:
        """
        Select the graphics item corresponding to the selected Model item
        :param index: the model index of the selected item (relative to proxy model)
        """
        source_index = self.proxy_model.mapToSource(index)
        # selected_model_index = self.proxy_model.mapToSource(index).sibling(source_index.row(), 0)
        selected_model_item = self.source_model.itemFromIndex(source_index.siblingAtColumn(0))
        if selected_model_item:
            self.object_tree_view.select_index(self.proxy_model.mapFromSource(source_index.siblingAtColumn(0)))
            # self.object_tree_view.select_index(selected_model_index)
            # self.object_tree_view.setCurrentIndex(self.proxy_model.mapFromSource(selected_model_index))
            obj = selected_model_item.object
            if QGuiApplication.mouseButtons() == Qt.LeftButton:
                if isinstance(obj, (BoundingBox, Point, Mask)):
                    self.scene.select_from_tree_view(obj.graphics_item)
                elif isinstance(obj, Object):
                    if self.source_model.get_bounding_box(obj, self.T) is not None:
                        self.scene.select_from_tree_view(self.source_model.get_bounding_box(obj, self.T).graphics_item)
            elif QGuiApplication.mouseButtons() == Qt.MiddleButton:
                _ = [r.setSelected(False) for r in self.scene.selectedItems()]
                if isinstance(obj, Object):
                    boxes, points, masks = self.source_model.get_all_prompt_items([obj], self.T)
                    for item in boxes + points + masks:
                        item.graphics_item.setSelected(True)
                else:
                    obj.graphics_item.setSelected(True)

    def create_video(self, video_dir: str) -> None:
        """
        Create a pseudo-video consisting of as many jpg files as there are frames. This pseudo-video will be fed info SAM2 for
        segmentation propagation from frame to frame
        :param video_dir: the destination directory for the jpg files
        """
        os.makedirs(video_dir, exist_ok=True)
        for frame in range(self.viewer.background.image.image_resource_data.sizeT):
            self.change_frame(frame)
            # self.viewer.background.image.change_frame(frame)
            self.viewer.background.image.pixmap().toImage().save(f'{video_dir}/{frame:05d}.jpg', format='jpg')

    def change_frame(self, T: int = 0) -> None:
        """
        Change the frame of the video
        :param T: the new frame time index
        """
        super().change_frame(T=T)
        self.redraw_scene()

    def redraw_scene(self) -> None:
        """
        Redraw the scene whenever something has changed (current frame, contour approximation method, fit ellipse)
        """
        self.scene.reset_graphics_items()
        boxes, points, masks = self.source_model.get_all_prompt_items(frame=self.T)
        for box in boxes:
            self.scene.addItem(box.graphics_item)
        for point in points:
            self.scene.addItem(point.graphics_item)
        for mask in masks:
            mask.set_graphics_item(self.contour_method)
            if self.display_ellipses.isChecked():
                self.scene.addItem(mask.ellipse_item)
            else:
                self.scene.addItem(mask.graphics_item)

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Slot activated when the segmentation tool is closed, mainly useful for releasing GPU memory
        :param event: the close event
        """
        print('Closing SegmentationTool')
        self.release_memory()

    def release_memory(self, keep_predictor: bool = False) -> None:
        """
        Releases GPU memory after processing

        :param keep_predictor: True if predictor should be kept alive for reuse, False otherwise
        """
        if keep_predictor is False and self.inference_state is not None:
            for v in self.inference_state.values():
                if torch.is_tensor(v):
                    del v
                    v = None
            self.predictor = None
            self.inference_state = None

        del self.video_segments
        del self.mask_logits
        self.video_segments = None
        self.mask_logits = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        gc.collect()

    def segment_from_prompt(self) -> None:
        """
        Gets the prompt specified using bounding boxes and points and segments objects accordingly
        """
        video_dir = os.path.join(get_config_value('project', 'workspace'),
                                 PyDetecDiv.project_name,
                                 'data/SegmentAnything2/videos',
                                 self.region)
        print(f'{video_dir=}')
        frame = self.T
        if not os.path.exists(video_dir):
            self.create_video(video_dir)
        self.change_frame(T=frame)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print(f'using device: {device}')

        if device.type == 'cuda':
            torch.autocast('cuda', dtype=torch.bfloat16).__enter__()

        sam2_checkpoint = get_config_value('paths', 'sam2_checkpoint')
        model_cfg = get_config_value('paths', 'sam2_model_cfg')

        if self.predictor is None:
            self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
            self.inference_state = self.predictor.init_state(video_dir)
        else:
            self.predictor.reset_state(self.inference_state)
            # self.inference_state = self.predictor.init_state(video_dir) # ??? should we init after resetting ?

        frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in ['.jpg', 'jpeg', '.JPG', '.JPEG']]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        present_objects = [obj for obj in self.source_model.objects if self.source_model.is_present_at_frame(obj, self.T)]
        # start_frame = min(self.source_model.get_entry_frame(obj) for obj in present_objects)
        start_frame = self.T
        end_frame = max(obj.exit_frame for obj in present_objects)

        for f1, f2 in self.key_frames_intervals():
            self.predictor.reset_state(self.inference_state)
            self.video_segments = {}
            f2 = min(f1 + self.max_frames_prop, self.viewer.image_resource_data.sizeT) if f2 == -1 else f2

            # print(f'{start_frame=} - {end_frame=}')
            if start_frame <= f1 < end_frame:
                print(f'{f2 - f1} frames starting from {f1} to {f2}')
                for obj_id, frames in self.source_model.get_prompt().items():
                    for frame, box_points in frames.items():
                        if frame < f2 and f1 < self.source_model.object(obj_id).exit_frame:
                            # print(f'adding {box_points=} for object {obj_id} at {frame=}')
                            _, out_obj_ids, self.mask_logits = self.predictor.add_new_points_or_box(
                                    inference_state=self.inference_state,
                                    frame_idx=frame,
                                    obj_id=obj_id,
                                    **box_points
                                    )

                for out_frame, out_obj_ids, self.mask_logits in self.predictor.propagate_in_video(self.inference_state,
                                                                                                  start_frame_idx=f1,
                                                                                                  max_frame_num_to_track=f2 - f1,
                                                                                                  reverse=False):
                    self.video_segments[out_frame] = {
                        out_obj_id: (self.mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                        }

                for out_frame in range(f1, f2):
                    for out_obj_id, out_mask in self.video_segments[out_frame].items():
                        mask_item = self.mask_to_shape(out_mask)

                        if mask_item is not None:
                            mask_item.setData(0, f'mask_{out_obj_id}')
                            self.source_model.set_mask(self.source_model.object(out_obj_id), out_frame, mask_item)
                            mask = self.source_model.get_mask(self.source_model.object(out_obj_id), out_frame)
                            mask.out_mask = out_mask[0]
                            mask.setBrush(QBrush(Colours.palette[int(out_obj_id) % len(Colours.palette)]))
            else:
                continue
        self.change_frame(self.T)
        self.proxy_model.invalidateFilter()
        self.release_memory(keep_predictor=True)

    def key_frames_intervals(self) -> list[list[int, int]]:
        """
        Returns intervals from key frame to key frame

        :return: the list of intervals
        """
        key_frames = self.source_model.all_key_frames

        intervals = []

        # Add intervals between successive elements
        intervals.extend([key_frames[i], key_frames[i + 1]] for i in range(len(key_frames) - 1))

        # Add the final interval if the last element is not equal to the max value
        if key_frames[-1] != self.viewer.image_resource_data.sizeT:
            intervals.append([key_frames[-1], -1])

        return intervals

    def mask_to_shape(self, mask: list) -> QGraphicsPolygonItem | None:
        """
        Creates graphics items from predicted masks

        :param mask: the mask
        :return: the ellipse approximation and the original polygon approximation of the mask
        """
        mask_shape = QPolygonF()
        contour = Mask.bitmap2contour(mask[0], self.contour_method)
        if contour is not None:
            for point in contour:
                mask_shape.append(QPointF(point[0][0], point[0][1]))
            if mask_shape.size() > 0:
                return QGraphicsPolygonItem(mask_shape)
        return None
