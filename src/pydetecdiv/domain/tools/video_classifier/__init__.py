"""
Video classifier tool
"""
import numpy as np
import tables
import torch
from torchvision.transforms import InterpolationMode, v2

from pydetecdiv.app.tools import Commands
from pydetecdiv.domain.tools.video_classifier.models import MViT, Swin3D, S3D, VideoResNet
from pydetecdiv.app.parameters import Parameters, IntParameter, FloatParameter, ChoiceParameter, StringParameter
from pydetecdiv.app.tools.deep_learning import ROIDataset, SupervisedDeepTool
from pydetecdiv.domain.tools.data import compute_class_weights
from pydetecdiv.domain.tools.video_classifier.models.VideoResNet import CustomR2Plus_1D
from pydetecdiv.domain.tools.video_classifier.train import VideoClassifierTrainer
from pydetecdiv.domain.tools.video_classifier.evaluate import VideoClassifierEvaluator
from pydetecdiv.domain.tools.video_classifier.predict import VideoClassifierPredictor
from pydetecdiv.domain.tools.data.hdf5 import ROIHDF5reader


class VideoClassifier(SupervisedDeepTool):
    """
    Video classifier tool, providing all functionalities for deep learning video classification (training, evaluation, prediction)
    """
    id_ = 'cnrs.plewniak.videoclassifier'
    version = '1.0.0'
    name = 'Video Classifier'

    def __init__(self, parameters: Parameters = Parameters(), commands: Commands = Commands(), working_dir: str | None = None,
                 device: torch.device | None = None, model: torch.nn.Module | None = None):
        super().__init__(parameters = parameters, commands=commands, working_dir=working_dir, device=device, model=model)

        self.parameters.update_parameters(
                commands={'train_model'},
                parameters = [
                    ChoiceParameter(name='model', label='Model name', default='CustomR2Plus_1D',
                                    items={'Swin3D_tiny'    : Swin3D.Swin3D_tiny,
                                           'Swin3D_small'   : Swin3D.Swin3D_small,
                                           'Swin3D_base'    : Swin3D.Swin3D_base,
                                           'S3D'            : S3D.S3D,
                                           'R3D_18'         : VideoResNet.R3D_18,
                                           'MC3_18'         : VideoResNet.MC3_18,
                                           'R2+1d_18'       : VideoResNet.R2Plus1d_18,
                                           'CustomR2Plus_1D': CustomR2Plus_1D,
                                           'MViT_small'     : MViT.MViT_v2_s,
                                           }),
                    StringParameter(name='layers', label='Blocks layers', default='[1, 2]'),
                    StringParameter(name='strides', label='Strides', default='[1, 2]'),
                    IntParameter(name='seq_len', label='Sequence length', maximum=16, default=4),
                    FloatParameter(name='dropout', label='Dropout', default=0.2, minimum=0.0, maximum=0.9),
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
