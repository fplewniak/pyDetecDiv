"""
Video classifier trainer class
"""
from typing import TYPE_CHECKING

import polars
import torch
from torchvision.transforms import v2

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers.plots import MatplotViewer
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
        training_dataset, validation_dataset = self.tool.prepare_data_for_training()

        print(f'Training dataset size: {len(training_dataset)}')
        print(f'Validation dataset size: {len(validation_dataset)}')

        img, target = training_dataset[0]
        roi_id, frame = training_dataset.get_ref(0)
        print(f'{roi_id}: {training_dataset.indices[0]}')
        print(training_dataset.roi(training_dataset.indices[0]['roi'].item()))
        print(img.shape, img[7].shape, target)

        seqlen = img.shape[0]
        plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=seqlen, rows=1)
        for i in range(seqlen):
            img_channel_last = (torch.as_tensor(img[i].permute([1, 2, 0])))/img[i].max()
            plot_viewer.axes[i].imshow(img_channel_last)
            if i == int(seqlen / 2):
                plot_viewer.axes[i].set_title(f'{training_dataset.class_names[target - 1]}')
            plot_viewer.axes[i].set_xlabel(f'{frame + i}')
        tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / {roi_id}')
        tab.project_name = PyDetecDiv.project_name
        tab.addTab(plot_viewer, 'Sample sequence')
        tab.setCurrentWidget(plot_viewer)

        training_dataset.close()
        validation_dataset.close()

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
