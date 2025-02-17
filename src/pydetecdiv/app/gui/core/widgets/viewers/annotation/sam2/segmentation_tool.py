import os

from PIL import Image as PILimage
from matplotlib import pyplot as plt

import numpy as np
import torch.cuda
from PySide6.QtWidgets import QGraphicsSceneMouseEvent, QMenuBar, QGraphicsRectItem, QMenu
from sam2.build_sam import build_sam2_video_predictor

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers import Scene
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer, VideoScene


class SegmentationScene(Scene):
    def contextMenuEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        The context menu

        :param event:
        """
        menu = QMenu()
        segment_action = menu.addAction('Run segmentation')
        segment_action.triggered.connect(
                lambda _: PyDetecDiv.main_window.active_subwindow.currentWidget().segment_from_prompt(self.items()))
        menu.exec(event.screenPos())


class SegmentationTool(VideoPlayer):
    """
    Annotator class extending the VideoPlayer class to define functionalities specific to ROI image annotation
    """

    def __init__(self, region_name: str):
        super().__init__()
        self.region = region_name
        self.run = None
        self.viewport_rect = None
        self.prompt = {}
        self.inference_state = None

    # def setup(self, menubar: QMenuBar = None) -> None:
    #     """
    #     Sets the Manual segmentation tool up
    #
    #     :param scene: the scene
    #     :param menubar: the menu bar
    #     """
    #     super().setup(menubar=menubar)
    #     # self.menubar.setup()

    def create_video(self):
        os.makedirs(os.path.join('/data3/SegmentAnything2/videos/', self.region), exist_ok=True)
        for frame in range(self.viewer.background.image.image_resource_data.sizeT):
            self.change_frame(frame)

            # print(f'/data3/SegmentAnything2/videos/{frame:05d}.jpg')
            self.viewer.background.image.pixmap().toImage().save(f'/data3/SegmentAnything2/videos/{self.region}/{frame:05d}.jpg', format='jpg')

    def segment_from_prompt(self, items):
        video_dir = os.path.join('/data3/SegmentAnything2/videos/', self.region)
        frame = self.T
        if not os.path.exists(video_dir):
            self.create_video()
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

        video_dir = os.path.join('/data3/SegmentAnything2/videos/', self.region)
        frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in ['.jpg', 'jpeg', '.JPG', '.JPEG']]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        self.inference_state = predictor.init_state(video_dir)

        boxes = [[i.x(), i.y(), i.x() + i.rect().width(),  i.y() + i.rect().height()] for i in items if isinstance(i, QGraphicsRectItem)]
        obj_ids = list(range(len(boxes)))
        print(obj_ids, boxes)

        for ann_obj_id, box in zip(obj_ids, boxes):
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state = self.inference_state,
                    frame_idx = self.T,
                    obj_id=ann_obj_id,
                    box=box,
                    )
        # image = PILimage.open(os.path.join(video_dir, frame_names[self.T]))
        # plt.figure(figsize=(9,6))
        # plt.title(f'frame {self.T}')
        # plt.imshow(image)
        # for i, obj_id in enumerate(out_obj_ids):
        #     show_box(boxes[i], plt.gca())
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
            plt.figure(figsize=(6,4))
            plt.title(f'frame {out_frame_idx}')
            plt.imshow(PILimage.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.show()


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color=np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap('tab10')
        cmap_idx = 0 if obj_id is None else obj_id + 1
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edge_color='white', linewidth=1.0)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edge_color='white', linewidth=1.0)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



