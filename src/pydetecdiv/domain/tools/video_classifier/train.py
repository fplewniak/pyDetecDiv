"""
Video classifier trainer class
"""
from typing import TYPE_CHECKING

import polars

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
            roi = project.get_object('ROI', 1)
            annotations = [dict(annotation.record()) for annotation in roi.annotations]
            print(annotations[0])
            print(annotations[0].keys())
            print(polars.from_records(annotations, schema=list(annotations[0].keys())))
            # classification = project.get_object('Classification', 1)
            # for run in classification.runs():
            #     print(f'Run: {run.id_} - {run.tool_name}/{run.command}')


    def training_loop(self):
        """
        The training loop for the video classifier, run once per epoch
        """
