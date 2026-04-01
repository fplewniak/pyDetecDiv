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

            # classification = project.get_object('Classification', 1)
            # for run in classification.runs():
            #     print(f'Run: {run.id_} - {run.tool_name}/{run.command}')


    def training_loop(self):
        """
        The training loop for the video classifier, run once per epoch
        """
