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
        self.tool.prepare_data_for_training()
        #run = self.tool.save_run(command='training')

        # with pydetecdiv_project(PyDetecDiv.project_name) as project:
        #     annotations_df = project.get_polars('RoiAnnotations')
        #     print(annotations_df)
            # roi_list = project.get_annotated_rois(ids_only=True, id_list=[2, 4, 6, 8])
            # print(roi_list)
            # all_annotations = project.get_objects('RoiAnnotations')
            # annotations = [ann for roi in project.get_annotated_rois() for ann in roi.annotations(as_records=True)]
            # print(polars.from_records(annotations, schema=list(annotations[0].keys())))
            # classification = project.get_object('Classification', 1)
            # for run in classification.runs():
            #     print(f'Run: {run.id_} - {run.tool_name}/{run.command}')


    def training_loop(self):
        """
        The training loop for the video classifier, run once per epoch
        """
