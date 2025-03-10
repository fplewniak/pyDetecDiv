import os
from pprint import pprint

from PIL import Image as PILimage
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QKeyEvent, QKeySequence
from matplotlib import pyplot as plt

import numpy as np
import torch.cuda
from PySide6.QtWidgets import (QGraphicsSceneMouseEvent, QMenu, QWidget, QGraphicsEllipseItem, QMenuBar, QVBoxLayout, QLabel,
                               QHBoxLayout, QSplitter, QGraphicsRectItem)
from sam2.build_sam import build_sam2_video_predictor

from pydetecdiv.app import PyDetecDiv, DrawingTools
from pydetecdiv.app.gui.core.widgets.viewers.annotation.sam2.objectsmodel import (ObjectsTreeView, Object, PromptProxyModel,
                                                                                  PromptSourceModel, ObjectReferenceRole,
                                                                                  BoundingBox)
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer, VideoViewerPanel, VideoControlPanel, VideoScene
from pydetecdiv.app.models.Trees import TreeItem


class SegmentationScene(VideoScene):
    def __init__(self, parent: QWidget = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.default_pen = QPen(Qt.GlobalColor.green, 1)
        self.positive_pen = QPen(Qt.GlobalColor.green, 1)
        self.negative_pen = QPen(Qt.GlobalColor.red, 1)
        self.pen = self.default_pen

        self.about_to_create_object = False
        self.object_list = []
        self.last_shape = None

    @property
    def current_object(self):
        return self.player.current_object

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

    def select_from_tree_view(self, graphics_item):
        _ = [r.setSelected(False) for r in self.selectedItems()]
        if graphics_item is not None:
            graphics_item.setSelected(True)

    def select_Item(self, event: QGraphicsSceneMouseEvent) -> None:
        graphics_item = super().select_Item(event)
        if isinstance(graphics_item, (QGraphicsRectItem, QGraphicsEllipseItem)):
            self.player.object_tree_view.select_object_from_graphics_item(graphics_item)

    def delete_item(self, r):
        if isinstance(r, QGraphicsRectItem):
            print('Removing bounding box for current object, but should actually select the object corresponding to the selected'
                  'QGraphicsItem')
            self.player.source_model.remove_bounding_box(self.current_object, self.player.T)
            # self.player.object_tree_view.select_item(self.player.current_object.tree_item)
            # self.player.source_model.delete_bounding_box(self.player.T, r)
            # print(f'removing item {r}')
            # self.removeItem(r)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            match PyDetecDiv.current_drawing_tool, event.modifiers():
                case DrawingTools.DrawRect, Qt.KeyboardModifier.NoModifier:
                    if self.last_shape:
                        if self.player.source_model.has_bounding_box(self.current_object, self.player.T):
                            self.player.source_model.change_bounding_box(self.current_object, self.player.T, self.last_shape)
                        else:
                            self.player.source_model.add_bounding_box(self.current_object, self.player.T, self.last_shape)
                        self.player.object_tree_view.expandAll()
                        self.last_shape = None
                case DrawingTools.DuplicateItem, Qt.KeyboardModifier.NoModifier:
                    if self.selectedItems():
                        print('Duplicating a bounding box')
                        rect_item = self.duplicate_selected_Item(event)
                        # self.player.prompt_model.add_bounding_box(self.current_object, self.player.T, rect_item)
                        # self.player.source_model.set_bounding_box(self.player.T, self.current_object, rect_item)
                        # self.player.object_tree_view.expandAll()

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
                case DrawingTools.DrawPoint, Qt.KeyboardModifier.ControlModifier:
                    self.add_point(event)

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

    # def move_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem | QGraphicsEllipseItem:
    #     self.last_shape = super().move_Item(event)
    #     return self.last_shape

    def draw_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem:
        if self.player.current_object is not None:
            self.last_shape = super().draw_Item(event)
        else:
            self.last_shape = None
        return self.last_shape
    #
    # def duplicate_selected_Item(self, event) -> QGraphicsRectItem | None:
    #     if self.player.current_object is not None:
    #         item = super().duplicate_selected_Item(event)
    #         return item
    #     return None

    # def contextMenuEvent(self, event: QGraphicsSceneMouseEvent) -> None:
    #     """
    #     The context menu
    #
    #     :param event:
    #     """
    #     menu = QMenu()
    #     segment_action = menu.addAction('Run segmentation')
    #     segment_action.triggered.connect(
    #             lambda _: self.player.segment_from_prompt(self.items()))
    #     menu.exec(event.screenPos())
    #
    # def new_object(self):
    #     self.player.prompt[f'{self.player.T}'] = {f'{self.object_count}': {'box': [], 'points': []}}
    #     self.current_object = self.object_count
    #     self.object_count += 1
    #     pprint(self.player.prompt)
    #
    # def add_point(self, event: QGraphicsSceneMouseEvent) -> QGraphicsEllipseItem | None:
    #     if self.current_object is not None:
    #         label = 1
    #         self.pen = self.positive_pen
    #         if event.buttons() == Qt.MouseButton.LeftButton:
    #             match PyDetecDiv.current_drawing_tool, event.modifiers():
    #                 case DrawingTools.DrawPoint, Qt.KeyboardModifier.ControlModifier:
    #                     self.pen = self.negative_pen
    #                     label = 0
    #         item = super().add_point(event)
    #         item.setData(1, label)
    #         self.pen = self.default_pen
    #         self.player.prompt[f'{self.player.T}'][f'{self.current_object}']['points'].append([item.data(0), item])
    #         pprint(self.player.prompt)
    #         return item
    #     return None
    #
    # def item_dict(self):
    #     item_dict = {'boxes': {}, 'points': {}}
    #     for item in sorted(self.regions(), key=lambda x: x.data(0)):
    #         self._add_item_to_dict(item, item_dict['boxes'])
    #     for item in sorted(self.points(), key=lambda x: x.data(0)):
    #         self._add_item_to_dict(item, item_dict['points'])
    #     return item_dict
    #
    # def draw_Item(self, event):
    #     item = super().draw_Item(event)
    #     item.setData(0, f'bounding_box_{item.x():.1f}_{item.y():.1f}')
    #     if f'{self.player.T}' not in self.player.prompt or self.current_object is None or item.data(0) not in \
    #             self.player.prompt[f'{self.player.T}'][f'{self.current_object}']:
    #         self.new_object()
    #         self.player.prompt[f'{self.player.T}'][f'{self.current_object}']['box'] = [item.data(0), item]
    #         pprint(self.player.prompt)
    #     return item
    #


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
        # self.prompt_model = PromptModel()
        self.inference_state = None
        self.object_tree_view = None

    # @property
    # def model(self):
    #     return self.object_tree_view.source_model

    @property
    def current_tree_index(self):
        return self.proxy_model.mapToSource(self.object_tree_view.currentIndex())

    @property
    def current_tree_item(self):
        return self.source_model.itemFromIndex(self.current_tree_index)

    @property
    def current_object(self):
        item = self.current_tree_item
        if item is not None:
            obj = item.data(ObjectReferenceRole)
            if isinstance(obj, Object):
                return obj
            if isinstance(obj, BoundingBox):
                return item.parent().data(ObjectReferenceRole)
        return None

    def add_object(self, obj):
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
        self.object_tree_view.setHeaderHidden(False)
        self.object_tree_view.setColumnHidden(1, True)
        self.object_tree_view.expandAll()
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

    def select_from_tree_view(self, index):
        selected_model_item = self.source_model.itemFromIndex(self.proxy_model.mapToSource(index))
        obj = selected_model_item.object
        if isinstance(obj, BoundingBox):
            self.scene.select_from_tree_view(obj.rect_item)
        elif isinstance(obj, Object):
            if self.source_model.get_bounding_box(obj, self.T) is not None:
                self.scene.select_from_tree_view(self.source_model.get_bounding_box(obj, self.T).rect_item)

    def create_video(self, video_dir):
        os.makedirs(video_dir, exist_ok=True)
        for frame in range(self.viewer.background.image.image_resource_data.sizeT):
            self.change_frame(frame)
            self.viewer.background.image.pixmap().toImage().save(f'{video_dir}/{frame:05d}.jpg', format='jpg')

    def segment_from_prompt(self, items):
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
