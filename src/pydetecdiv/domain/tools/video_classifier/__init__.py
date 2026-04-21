"""
Video classifier tool
"""
import datetime
import locale

import numpy as np
import polars
import tables
import torch
from torch import optim
from torchvision.transforms import InterpolationMode, v2

from pydetecdiv.app.parameters import Parameters, IntParameter, FloatParameter, ChoiceParameter, PathParameter, CheckParameter
from pydetecdiv.app.tools.deep_learning import DeepTool, ModelTrainer, ModelEvaluator, Predictor, ROIDataset
from pydetecdiv.domain.tools.data import compute_class_weights
from pydetecdiv.domain.tools.video_classifier.train import VideoClassifierTrainer
from pydetecdiv.domain.tools.video_classifier.evaluate import VideoClassifierEvaluator
from pydetecdiv.domain.tools.video_classifier.predict import VideoClassifierPredictor
from pydetecdiv.domain.tools.data.hdf5 import ROIHDF5reader
from pydetecdiv.utils.Alphabets import greek


class VideoClassifier(DeepTool):
    """
    Video classifier tool, providing all functionalities for deep learning video classification (training, evaluation, prediction)
    """
    id_ = 'cnrs.plewniak.videoclassifier'
    version = '1.0.0'
    name = 'Video Classifier'

    def __init__(self, parameters: Parameters | None = None, working_dir: str | None = None):
        super().__init__(parameters, working_dir)
        self.parameters = Parameters(
                [
                    IntParameter(name='epochs', label='Epochs', default=32),
                    IntParameter(name='batch_size', label='Batch size', default=8, ),
                    ChoiceParameter(name='optimizer', label='Optimizer', default='AdamW',
                                    items={'AdamW'   : optim.AdamW,
                                           'SGD'     : optim.SGD,
                                           'Adadelta': optim.Adadelta,
                                           'Adamax'  : optim.Adamax,
                                           'Nadam'   : optim.NAdam,
                                           }, ),
                    IntParameter(name='seed', label='Random seed', maximum=999999999, default=42, ),
                    FloatParameter(name='learning_rate', label='Learning rate', default=1.5e-4, minimum=1e-20, maximum=1.0),
                    FloatParameter(name='focal_gamma', label='Focal loss gamma', default=1.0, minimum=0.0, maximum=2.0, ),
                    FloatParameter(name='num_training', label='Training dataset', default=0.4, minimum=0.01, maximum=0.98, ),
                    FloatParameter(name='num_validation', label='Validation dataset', default=0.3, minimum=0.01, maximum=0.98, ),
                    FloatParameter(name='num_test', label='Test dataset', default=0.3, minimum=0.01, maximum=0.98, enabled=False),
                    IntParameter(name='data_seed', label='Random seed', maximum=999999999, default=42),
                    PathParameter(name='hdf5_file', label='', select_dir=False, filters=["HDF5 (*.h5 *.hdf5)", ],
                                  default='roi_data.h5', ),
                    CheckParameter(name='time_first', label='Time first', default=False),
                    CheckParameter(name='augmentation', label='Augmentation', groups={'training', 'finetune'}, default=False,
                                   exclusive=False),
                    ChoiceParameter(name='regularization', label='Regularization method', default='Ridge (L2)',
                                    items={'None'      : 0,
                                           'LASSO (L1)': 1,
                                           'Ridge (L2)': 2,
                                           }),
                    FloatParameter(name='lambda_reg', label=f'{greek["lambda"]} parameter', default=2e-5, ),
                    # IntParameter(name='idx', label='Dataset index', maximum=999999999, minimum=0, default=0),
                    ]
                )

    def prepare_data_for_training(self, image_shape=(224, 224), *args, **kwargs) -> tuple[ROIDataset, ROIDataset, torch.Tensor]:
        """
        Prepare the data for training
        """
        hdf5_reader = ROIHDF5reader(tables.open_file(self.parameters.hdf5_file.value, mode='r'),
                                    time_first=self.parameters.time_first.value)
        roi_idx = list(range(hdf5_reader.num_rois))
        np.random.default_rng(self.parameters.data_seed.value)
        np.random.shuffle(roi_idx)

        num_training = int(hdf5_reader.num_rois * self.parameters.num_training + 0.5)
        num_validation = int(hdf5_reader.num_rois * self.parameters.num_validation + 0.5)

        training_idx = hdf5_reader.target_indices(roi_idx[:num_training])
        validation_idx = hdf5_reader.target_indices(roi_idx[num_training:num_training + num_validation])
        class_weights = compute_class_weights(hdf5_reader.targets)

        hdf5_reader.close()

        augmentation = v2.RandomAffine(degrees=5.0, translate=(4.0 / 60.0, 4.0 / 60.0), scale=(0.9, 1.111),
                                       interpolation=InterpolationMode.BILINEAR) if self.parameters.augmentation else None
        print(f'{augmentation=}')

        training_dataset = ROIDataset(ROIHDF5reader(tables.open_file(self.parameters.hdf5_file.value, mode='r'),
                                                    time_first=self.parameters.time_first.value),
                                      training_idx, targets=True, image_shape=image_shape, transform=augmentation)
        validation_dataset = ROIDataset(ROIHDF5reader(tables.open_file(self.parameters.hdf5_file.value, mode='r'),
                                                      time_first=self.parameters.time_first.value),
                                        validation_idx, targets=True, image_shape=image_shape)

        return training_dataset, validation_dataset, class_weights

    def prepare_data_for_prediction(self, *args, **kwargs) -> None:
        """
        Prepare the data for prediction
        """

    def create_trainer(self) -> VideoClassifierTrainer:
        """
        Concrete factory to create video classifier trainer
        """
        return VideoClassifierTrainer(self)

    def create_evaluator(self) -> VideoClassifierEvaluator:
        """
        Concrete factory to create video classifier evaluator
        """
        return VideoClassifierEvaluator(self)

    def create_predictor(self) -> VideoClassifierPredictor:
        """
        Concrete factory to create video classifier predictor
        """
        return VideoClassifierPredictor(self)

    # def save_run(self, *args, **kwargs):
    #     """
    #     Concrete method saving the video classifier run
    #     """
