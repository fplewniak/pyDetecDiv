"""
Computer Assisted Segmentation Tool: a tool for manual segmentation of images (annotation) using SegmentAnything2 by META to
segment and propagate segmentation using prompts (bounding boxes, points and masks)
"""
import os
from pprint import pprint

from PIL import Image as PILimage
from PySide6.QtCore import Qt, QModelIndex
from PySide6.QtGui import QPen, QKeyEvent, QKeySequence, QStandardItem
from matplotlib import pyplot as plt

import numpy as np
import torch.cuda
from PySide6.QtWidgets import (QGraphicsSceneMouseEvent, QMenu, QWidget, QGraphicsEllipseItem, QMenuBar, QVBoxLayout, QLabel,
                               QHBoxLayout, QSplitter, QGraphicsRectItem, QHeaderView, QGraphicsItem)
from sam2.build_sam import build_sam2_video_predictor

from pydetecdiv.app import PyDetecDiv, DrawingTools
from pydetecdiv.app.gui.core.widgets.viewers.annotation.sam2.objectsmodel import (ObjectsTreeView, Object, PromptProxyModel,
                                                                                  PromptSourceModel, ObjectReferenceRole,
                                                                                  BoundingBox, Point, ModelItem)
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer, VideoViewerPanel, VideoControlPanel, VideoScene


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
            if isinstance(item, (QGraphicsRectItem, QGraphicsEllipseItem)):
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
            if isinstance(graphics_item, (QGraphicsRectItem, QGraphicsEllipseItem)):
                self.player.object_tree_view.select_object_from_graphics_item(graphics_item)

    def select_Item(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        method overriding the select_Item method of QGraphicsScene in order to synchronize the selection in the Scene with a
        selection in the tree view.
        :param event:
        """
        graphics_item = super().select_Item(event)
        if isinstance(graphics_item, (QGraphicsRectItem, QGraphicsEllipseItem)):
            self.player.object_tree_view.select_object_from_graphics_item(graphics_item)

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
        frame_segment_action = menu.addAction('Run segmentation on current frame')
        frame_segment_action.triggered.connect(lambda _: pprint(self.player.prompt))
        video_segment_action = menu.addAction('Run segmentation on video')
        video_segment_action.triggered.connect(lambda _: pprint(self.player.source_model.get_prompt()))
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
        self.inference_state = None
        self.object_tree_view = None

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
        return new_item

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
        self.object_tree_view.clicked.connect(self.select_from_tree_view)

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
        self.time_display.setGeometry(20, 30, 140, self.time_display.height())

        self.video_frame.connect(self.proxy_model.set_frame)

    def select_from_tree_view(self, index: QModelIndex) -> None:
        """
        Select the graphics item corresponding to the selected Model item
        :param index: the model index of the selected item (relative to proxy model)
        """
        source_index = self.proxy_model.mapToSource(index)
        selected_model_index = self.proxy_model.mapToSource(index).sibling(source_index.row(), 0)
        selected_model_item = self.source_model.itemFromIndex(selected_model_index)
        if selected_model_item:
            self.object_tree_view.setCurrentIndex(self.proxy_model.mapFromSource(selected_model_index))
        obj = selected_model_item.object
        if isinstance(obj, BoundingBox):
            self.scene.select_from_tree_view(obj.graphics_item)
        elif isinstance(obj, Point):
            self.scene.select_from_tree_view(obj.graphics_item)
        elif isinstance(obj, Object):
            if self.source_model.get_bounding_box(obj, self.T) is not None:
                self.scene.select_from_tree_view(self.source_model.get_bounding_box(obj, self.T).graphics_item)

    def create_video(self, video_dir: str) -> None:
        """
        Create a pseudo-video consisting of as many jpg files as there are frames. This pseudo-video will be fed info SAM2 for
        segmentation propagation from frame to frame
        :param video_dir: the destination directory for the jpg files
        """
        os.makedirs(video_dir, exist_ok=True)
        for frame in range(self.viewer.background.image.image_resource_data.sizeT):
            self.change_frame(frame)
            self.viewer.background.image.pixmap().toImage().save(f'{video_dir}/{frame:05d}.jpg', format='jpg')

    def change_frame(self, T: int = 0) -> None:
        """
        Change the frame of the video
        :param T: the new frame time index
        """
        super().change_frame(T=T)
        self.scene.reset_graphics_items()
        boxes, points = self.source_model.get_all_prompt_items(self.T)
        for box in boxes:
            self.scene.addItem(box.graphics_item)
        for point in points:
            self.scene.addItem(point.graphics_item)

    def segment_from_prompt(self, items):
        """

        :param items:
        """
        video_dir = os.path.join('/data3/SegmentAnything2/videos/', self.region)
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

        sam2_checkpoint = '/data3/SegmentAnything2/checkpoints/sam2.1_hiera_large.pt'
        model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'

        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

        frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in ['.jpg', 'jpeg', '.JPG', '.JPEG']]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        self.inference_state = predictor.init_state(video_dir)

        # boxes = [[i.x(), i.y(), i.x() + i.rect().width(), i.y() + i.rect().height()] for i in items if
        #          isinstance(i, QGraphicsRectItem)]
        # obj_ids = list(range(len(boxes)))
        # print(obj_ids, boxes)
        # for ann_obj_id, box in zip(obj_ids, boxes):
        #     _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        #             inference_state=self.inference_state,
        #             frame_idx=self.T,
        #             obj_id=ann_obj_id,
        #             box=box,
        #             )
        #
        # image = PILimage.open(os.path.join(video_dir, frame_names[self.T]))
        # plt.figure(figsize=(9, 6))
        # plt.title(f'frame {self.T}')
        # plt.imshow(image)
        # for i, obj_id in enumerate(out_obj_ids):
        #     show_box(boxes[i], plt.gca())
        #     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=obj_id)
        # plt.show()

        points = [[[i.x(), i.y()]] for i in items if isinstance(i, QGraphicsEllipseItem)]
        labels = [[i.data(0)] for i in items if isinstance(i, QGraphicsEllipseItem)]
        obj_ids = list(range(len(points)))

        prompts = {}
        for ann_obj_id, pts, lbls in zip(obj_ids, points, labels):
            prompts[ann_obj_id] = pts, lbls
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.T,
                    obj_id=ann_obj_id,
                    points=pts,
                    labels=lbls
                    )

        # image = PILimage.open(os.path.join(video_dir, frame_names[self.T]))
        # plt.figure(figsize=(9,6))
        # plt.title(f'frame {self.T}')
        # plt.imshow(image)
        # # show_points(pts, lbls, plt.gca())
        # for i, obj_id in enumerate(out_obj_ids):
        #     # show_points(*prompts[obj_id], plt.gca())
        #     show_mask((out_mask_logits[i]>0.0).cpu().numpy(), plt.gca(), obj_id=obj_id)
        # plt.show()

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
                }
        vis_frame_stride = 3
        num_frames = 5
        plt.close('all')
        for out_frame_idx in range(0, num_frames * vis_frame_stride, vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f'frame {out_frame_idx}')
            plt.imshow(PILimage.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.show()


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap('tab10')
        cmap_idx = 0 if obj_id is None else obj_id + 1
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=20):
    print(coords, labels)
    pos_points = coords[labels == 1]
    print('Pos:', pos_points)
    neg_points = coords[labels == 0]
    print('Neg:', neg_points)
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edge_color='white', linewidth=1.0)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edge_color='white', linewidth=1.0)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
