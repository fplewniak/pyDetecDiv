"""
Video classifier tool
"""
import datetime
import locale
import sys

import numpy as np
import polars
import tables
import torch
from torch import optim
from torchvision.transforms import InterpolationMode, v2

from pydetecdiv.domain.tools.video_classifier.models import MViT, Swin3D, S3D, VideoResNet
from pydetecdiv.app.parameters import (Parameters, IntParameter, FloatParameter, ChoiceParameter, PathParameter, CheckParameter,
                                       StringParameter)
from pydetecdiv.app.tools.deep_learning import DeepTool, ModelTrainer, ModelEvaluator, Predictor, ROIDataset
from pydetecdiv.domain.tools.data import compute_class_weights
from pydetecdiv.domain.tools.video_classifier.models.VideoResNet import CustomR2Plus_1D
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
                    ChoiceParameter(name='model', label='Model name', default='R2+1d_18',
                                    items={'Swin3D_tiny'    : Swin3D.Swin3D_tiny,
                                           'Swin3D_small'   : Swin3D.Swin3D_small,
                                           'Swin3D_base'    : Swin3D.Swin3D_base,
                                           'S3D'            : S3D.S3D,
                                           'R3D_18'         : VideoResNet.R3D_18,
                                           'MC3_18'         : VideoResNet.MC3_18,
                                           'R2+1d_18'       : VideoResNet.R2Plus1d_18,
                                           'CustomR2Plus_1D': CustomR2Plus_1D,
                                           }, commands={'train_model'}),
                    StringParameter(name='layers', label='Blocks layers', default='[1, 2]', commands={'train_model'}),
                    StringParameter(name='strides', label='Strides', default='[1, 2]', commands={'train_model'}),
                    IntParameter(name='epochs', label='Epochs', default=32, commands={'train_model'}),
                    IntParameter(name='batch_size', label='Batch size', default=8, commands={'train_model'}),
                    ChoiceParameter(name='optimizer', label='Optimizer', default='AdamW',
                                    items={'AdamW'   : optim.AdamW,
                                           'SGD'     : optim.SGD,
                                           'Adadelta': optim.Adadelta,
                                           'Adamax'  : optim.Adamax,
                                           'Nadam'   : optim.NAdam,
                                           }, commands={'train_model'}),
                    IntParameter(name='seed', label='Random seed', maximum=999999999, default=42, commands={'train_model'}),
                    FloatParameter(name='learning_rate', label='Learning rate', default=1.0e-4, minimum=1e-20, maximum=1.0,
                                   commands={'train_model'}),
                    FloatParameter(name='focal_gamma', label=f'Focal loss {greek["gamma"]}', default=1.5, minimum=0.0, maximum=2.0,
                                   commands={'train_model'}),
                    FloatParameter(name='num_training', label='Training dataset', default=0.4, minimum=0.01, maximum=0.98,
                                   commands={'train_model'}),
                    FloatParameter(name='num_validation', label='Validation dataset', default=0.3, minimum=0.01, maximum=0.98,
                                   commands={'train_model'}),
                    FloatParameter(name='num_test', label='Test dataset', default=0.3, minimum=0.01, maximum=0.98,
                                   commands={'train_model'}),
                    IntParameter(name='data_seed', label='Random seed', maximum=999999999, default=42,
                                 commands={'train_model'}),
                    PathParameter(name='hdf5_file', label='', select_dir=False, filters=["HDF5 (*.h5 *.hdf5)"],
                                  default='roi_data.h5', commands={'train_model'}),
                    CheckParameter(name='time_first', label='Time first', default=False, commands={'train_model'}),
                    CheckParameter(name='augmentation', label='Augmentation', default=False, exclusive=False,
                                   commands={'train_model'}),
                    IntParameter(name='seq_len', label='Sequence length', maximum=14, default=4, commands={'train_model'}),
                    ChoiceParameter(name='regularization', label='Regularization method', default='LASSO (L1)',
                                    items={'None'      : 0,
                                           'LASSO (L1)': 1,
                                           'Ridge (L2)': 2,
                                           }, commands={'train_model'}),
                    FloatParameter(name='lambda_reg', label=f'{greek["lambda"]} parameter', default=2e-5, minimum=1e-8,
                                   maximum=10.0, commands={'train_model'}),
                    FloatParameter(name='dropout', label='Dropout', default=0.2, minimum=0.0, maximum=0.9,
                                   commands={'train_model'}),
                    CheckParameter(name='warmup', label='Warm-up', default=False, exclusive=False, commands={'train_model'}),
                    FloatParameter(name='wu_start', label='   * warm-up start factor', default=0.1, minimum=0.1, maximum=0.5,
                                   commands={'train_model'}),
                    FloatParameter(name='wu_end', label='   * warm-up end factor', default=1.0, minimum=0.5, maximum=1.0,
                                   commands={'train_model'}),
                    IntParameter(name='wu_duration', label='   * warm-up duration', default=8, minimum=2, maximum=100,
                                 commands={'train_model'}),
                    CheckParameter(name='step_scheduler', label='Step scheduler', default=True, exclusive=True,
                                   commands={'train_model'}),
                    FloatParameter(name='step_gamma', label=f'   * {greek["gamma"]} parameter', default=0.95, minimum=0.01,
                                   maximum=0.99, commands={'train_model'}),
                    IntParameter(name='step_size', label='   * step size', default=4, minimum=1, maximum=100,
                                 commands={'train_model'}),
                    CheckParameter(name='reduce_lr_on_plateau', label='Reduce LR on plateau', default=False, exclusive=False,
                                   commands={'train_model'}),
                    IntParameter(name='reduce_patience', label='   * patience', default=10, minimum=1, maximum=100,
                                 commands={'train_model'}),
                    FloatParameter(name='reduction_factor', label='   * reduction factor', default=0.5, minimum=0.1, maximum=1.0,
                                   commands={'train_model'}),
                    # IntParameter(name='idx', label='Dataset index', maximum=999999999, minimum=0, default=0),
                    ]
                )

    def prepare_data_for_training(self, *args, image_shape=(224, 224), **kwargs) -> tuple[ROIDataset, ROIDataset, torch.Tensor]:
        """
        Prepare the data for training
        """
        # slice_seq = slice(0, 14) if self.parameters.model.key == 'S3D' else slice(4, 8)
        slice_seq = slice(int(7 - self.parameters.seq_len / 2), int(7 + self.parameters.seq_len / 2))
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

        augmentation = v2.RandomAffine(degrees=5.0, translate=(0.08, 0.08), scale=(0.9, 1.111),
                                       interpolation=InterpolationMode.BILINEAR) if self.parameters.augmentation else None
        print(f'{augmentation=}')

        training_dataset = ROIDataset(ROIHDF5reader(tables.open_file(self.parameters.hdf5_file.value, mode='r'),
                                                    time_first=self.parameters.time_first.value),
                                      training_idx, targets=True, image_shape=image_shape, slice_seq=slice_seq,
                                      transform=augmentation)
        validation_dataset = ROIDataset(ROIHDF5reader(tables.open_file(self.parameters.hdf5_file.value, mode='r'),
                                                      time_first=self.parameters.time_first.value),
                                        validation_idx, targets=True, image_shape=image_shape, slice_seq=slice_seq)

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
