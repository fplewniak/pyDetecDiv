"""
Video classifier trainer class
"""
import time
from typing import TYPE_CHECKING

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.app.tools.deep_learning import ModelTrainer

if TYPE_CHECKING:
    from pydetecdiv.domain.tools.video_classifier import VideoClassifier


class VideoClassifierTrainer(ModelTrainer):
    """
    Video classifier trainer class to run the training of deep learning video classifier model
    """
    def __init__(self, tool: 'VideoClassifier'):
        super().__init__(tool)

    def train_model(self):
        """
        Train the video classifier model, running the training loop once per epoch for as many epochs as requested by the user
        """
        print("Training video classifier model...")
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            fov = project.get_object('FOV', 1)
            start = time.perf_counter()
            print(fov)
            print(fov.image_resource().shape)
            image_resource_data = fov.image_resource().image_resource_data()
            for roi in fov.roi_list:
                start_partiel = time.perf_counter()
                (x1, y1), (x2, y2) = (roi.top_left, roi.bottom_right)
            # print(torch.tensor(np.array([image_resource_data.image(C=0, T=t, Z=0, sliceX=slice(x1, x2+1), sliceY=slice(y1, y2+1)) for t in range(15)])).shape)
                for t in range(image_resource_data.sizeT - 15):
                    _ = image_resource_data.sequence(15, T=t, crop=(slice(x1, x2+1), slice(y1, y2+1)))
                print(f'{roi.name}: {time.perf_counter() - start_partiel}')
            print(time.perf_counter() - start)

    def training_loop(self):
        """
        The training loop for the video classifier, run once per epoch
        """
