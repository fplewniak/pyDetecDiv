"""
Abstract DeepTool class
"""
import gc
import os
import pickle
import sys
from abc import abstractmethod
from typing import Iterable

import polars
import torch
from torch import Tensor, optim
from torch.optim.lr_scheduler import SequentialLR, StepLR, ReduceLROnPlateau, LinearLR, LRScheduler
from torch.utils.data import Dataset
from torchvision.transforms import v2, transforms

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv, set_connections
from pydetecdiv.app.gui.core.widgets.viewers.plots import MatplotViewer
from pydetecdiv.app.parameters import Parameters, IntParameter, ChoiceParameter, FloatParameter, CheckParameter, FileParameter
from pydetecdiv.app.tools import Tool

from pydetecdiv.app.tools.deep_learning.train import ModelTrainer
from pydetecdiv.app.tools.deep_learning.evaluate import ModelEvaluator
from pydetecdiv.app.tools.deep_learning.predict import Predictor
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.Run import Run
from pydetecdiv.domain.tools.data import RoiDataReader
from pydetecdiv.torch import TrainingStats
from pydetecdiv.torch.transforms import toStandardizedFloat32
from pydetecdiv.utils.Alphabets import greek


def find_tensors_on_gpu():
    """
    Find tensors stored on GPU (for debugging purposes)
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f"Tensor: Size: {obj.size()}, Device: {obj.device}", file=sys.stderr)
        except:
            pass


def find_gpu_tensor_references():
    """
    Find references to tensors stored on GPU (for debugging purposes)
    """
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            print(f"\nTensor: {obj}, Size: {obj.size()}, Device: {obj.device}", file=sys.stderr)
            referrers = gc.get_referrers(obj)
            print(f"  Referrers: {len(referrers)}", file=sys.stderr)
            for ref in referrers:
                if isinstance(ref, dict):
                    for k, v in ref.items():
                        if v is obj:
                            print(f"    - Dict key: {k}", file=sys.stderr)
                elif isinstance(ref, list):
                    for i, item in enumerate(ref):
                        if item is obj:
                            print(f"    - List index: {i}", file=sys.stderr)
                elif hasattr(ref, '__dict__'):
                    for attr, val in ref.__dict__.items():
                        if val is obj:
                            print(f"    - Attribute '{attr}' of {type(ref)}", file=sys.stderr)
                elif not isinstance(ref, type) and not isinstance(ref, str):
                    print(f"    - {type(ref)}", file=sys.stderr)


def set_optimizer(parameters: Parameters, model_param: dict | Iterable) -> torch.optim.Optimizer:
    """
    Set the optimizer.

    :param parameters: the parameters
    :param model_param: model parameters that will be passed to the optimizer constructor
    :return: the optimizer
    """
    lr = parameters.learning_rate.value if 'learning_rate' in parameters else 0.001
    weight_decay = parameters.weight_decay.value if 'weight_decay' in parameters else 0.01
    momentum = parameters.momentum.value if 'momentum' in parameters else 0.9
    optimizer = parameters.optimizer.value(model_param, lr=lr, weight_decay=weight_decay)
    match parameters.optimizer.key:
        case 'SGD':
            optimizer = parameters.optimizer.value(model_param, lr=lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer


def set_schedulers(parameters: Parameters, optimizer: torch.optim.Optimizer) -> tuple[LRScheduler, ReduceLROnPlateau | None]:
    """
    Set the schedulers for adjusting learning rate during training

    :param parameters: the tool parameters
    :param optimizer: the optimizer
    :return: a tuple with the main scheduler and the optional ReduceLROnPlateau
    """
    reduce_on_plateau = None
    main_scheduler = StepLR(optimizer, step_size=parameters.step_size.value, gamma=parameters.step_gamma.value, last_epoch=-1)
    if parameters.step_scheduler:
        print('Step scheduler', file=sys.stderr)

    if parameters.warmup:
        print('Warm-up scheduler', file=sys.stderr)
        warmup = LinearLR(optimizer, start_factor=parameters.wu_start.value, end_factor=parameters.wu_end.value,
                          total_iters=parameters.wu_duration.value)
        main_scheduler = SequentialLR(optimizer, schedulers=[warmup, main_scheduler], milestones=[parameters.wu_duration.value])

    if parameters.reduce_lr_on_plateau:
        print('Reduce LR on plateau scheduler', file=sys.stderr)
        reduce_on_plateau = ReduceLROnPlateau(optimizer, mode='min', patience=parameters.reduce_patience.value,
                                              factor=parameters.reduction_factor.value)

    return main_scheduler, reduce_on_plateau


class ROIDataset(Dataset):
    """
    A Pytorch dataset for ROI access in deep learning tools.
    """

    def __init__(self, data_reader: RoiDataReader, indices: polars.DataFrame, targets: bool = False,
                 image_shape: tuple[int, int] = (60, 60), slice_seq: slice | None = None, transform: torch.nn.Module | None = None):
        self.reader = data_reader
        self.indices = indices
        self.targets = targets
        self.image_shape = list(image_shape)
        self.slice = slice_seq
        self.transform = transforms.Compose([v2.Resize(image_shape), toStandardizedFloat32()])
        if transform:
            self.transform = transforms.Compose([self.transform, transform])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor, Tensor]:
        df = self.indices[idx]
        roi_idx = df['roi'].item()
        frame_idx = df['frame'].item()
        item = self.reader.roi_data(roi_idx=roi_idx, frame=frame_idx)
        if len(item.shape) == 4:
            item = item[self.slice, :]
        if self.transform:
            item = self.transform(item)
        if self.targets:
            target = self.reader.target(roi_idx=roi_idx, frame=frame_idx)
            return item, target
        return item

    def close(self):
        """
        Close the reader
        """
        self.reader.close()

    @property
    def class_names(self) -> list[str]:
        """
        Return the class names for the corresponding classification

        :return: the list of class names
        """
        return self.reader.class_names

    def roi(self, idx: int) -> ROI:
        """
        Return the ROI for the given index.

        :param idx: the index
        :return: the ROI object
        """
        return self.reader.roi(idx)

    def get_ref(self, idx: int) -> tuple[int, int]:
        """
        Get the references (roi and frame indices) of the ROIDataset item

        :param idx: the dataset index
        :return: tuple with the ROI.id_ and the frame index
        """
        df = self.indices[idx]
        roi_idx = df['roi'].item()
        frame_idx = df['frame'].item()
        return self.reader.roi_id(roi_idx), frame_idx

    def plot_sample(self, idx: int) -> None:
        """
        Plot the sample having index = idx

        :param idx: the index of the sample
        """
        sequence, target = self[idx]
        print(sequence.shape)
        roi_id, frame = self.get_ref(idx)
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            roi = project.get_object('ROI', roi_id)

        rowlen = 5
        plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=rowlen, rows=3)
        for i in range(3):
            for j in range(rowlen):
                img_channel_last = torch.as_tensor(sequence[rowlen * i + j].permute([1, 2, 0]))
                plot_viewer.axes[i][j].imshow(img_channel_last)
                if (rowlen * i + j) == int(3 * rowlen / 2):
                    plot_viewer.axes[i][j].set_title(f'{self.class_names[target]}')
                plot_viewer.axes[i][j].set_xlabel(f'{frame + rowlen * i + j}')
        tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / {roi.name}')
        tab.project_name = PyDetecDiv.project_name
        tab.addTab(plot_viewer, 'Sample sequence')
        tab.setCurrentWidget(plot_viewer)


class DeepTool(Tool):
    """
    DeepTool abstract class providing the basic functionality for deep-learning new_tools
    """

    def __init__(self, parameters: Parameters = Parameters(), working_dir: str | None = None, device: torch.device | None = None,
                 model: torch.nn.Module | None = None):
        super().__init__(parameters=parameters, working_dir=working_dir)
        self.parameters.update_parameters(
                [
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
                    ChoiceParameter(name='regularization', label='Regularization method', default='LASSO (L1)',
                                    items={'None'      : 0,
                                           'LASSO (L1)': 1,
                                           'Ridge (L2)': 2,
                                           }, commands={'train_model'}),
                    FloatParameter(name='lambda_reg', label=f'{greek["lambda"]} parameter', default=2e-5, minimum=1e-8,
                                   maximum=10.0, commands={'train_model'}),
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
                    ]
                )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = model
        if self.model is not None:
            self.model.to(self.device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = None, None, None
        self.dataloader = None
        self._model_trainer = None
        self._model_evaluator = None
        self._model_predictor = None

    def checkpoints_path(self, run: Run) -> str:
        """
        Return the path where to store checkpoints achieved when training a model

        :param run: the training run
        :return: the path
        """
        path = os.path.join(self.run_path(run), 'checkpoints')
        os.makedirs(path, exist_ok=True)
        return path

    def dump_train_stats(self, train_stats: TrainingStats):
        """
        Dumpt training statistics object into a pickle file
        :param train_stats:
        """
        if self.run is not None:
            train_stats_filepath = os.path.join(self.run_path(self.run), 'train_stats.pckl')
            with open(train_stats_filepath, 'wb') as fp:
                pickle.dump(train_stats, fp, protocol=pickle.HIGHEST_PROTOCOL)
        del train_stats
        gc.collect()
        torch.cuda.empty_cache()

    def set_model(self, model: torch.nn.Module):
        """
        Set the deep-learning model used by the tool

        :param model: the deep learning model
        """
        self.model = model
        self.model.to(self.device)

    @property
    def model_trainer(self) -> 'ModelTrainer':
        """
        Return the model trainer object associated with this tool

        :return: the model trainer
        """
        if self._model_trainer is None:
            self._model_trainer = self.create_trainer()
        return self._model_trainer

    @property
    def model_evaluator(self) -> 'ModelEvaluator':
        """
        Return the model evaluator object associated with this tool

        :return: the model evaluator
        """
        if self._model_evaluator is None:
            self._model_evaluator = self.create_evaluator()
        return self._model_evaluator

    @property
    def model_predictor(self) -> 'Predictor':
        """
        Return the model predictor object associated with this tool

        :return: the model predictor
        """
        if self._model_predictor is None:
            self._model_predictor = self.create_predictor()
        return self._model_predictor

    @abstractmethod
    def create_trainer(self) -> 'ModelTrainer':
        """
        Abstract factory method to create the model trainer
        """

    @abstractmethod
    def create_evaluator(self) -> 'ModelEvaluator':
        """
        Abstract factory method to create the model evaluator
        """

    @abstractmethod
    def create_predictor(self) -> 'Predictor':
        """
        Abstract factory method to create the model predictor
        """

    @abstractmethod
    def prepare_data_for_training(self, *args, **kwargs) -> tuple[ROIDataset, ROIDataset, torch.Tensor]:
        """
        Abstract method to prepare the data for training
        """

    @abstractmethod
    def prepare_data_for_prediction(self, *args, **kwargs) -> None:
        """
        Abstract method to prepare the data for prediction
        """


class SupervisedDeepTool(DeepTool):
    """
    Generic class defining tool for supervised deep learning
    """
    def __init__(self, parameters: Parameters = Parameters(), working_dir: str | None = None, device: torch.device | None = None,
                 model: torch.nn.Module | None = None):
        super().__init__(parameters=parameters, working_dir=working_dir)
        self.parameters.update_parameters(
                [FloatParameter(name='num_training', label='Training dataset', default=0.4, minimum=0.01, maximum=0.98,
                                   commands={'train_model'}),
                    FloatParameter(name='num_validation', label='Validation dataset', default=0.3, minimum=0.01, maximum=0.98,
                                   commands={'train_model'}),
                    FloatParameter(name='num_test', label='Test dataset', default=0.3, minimum=0.01, maximum=0.98,
                                   commands={'train_model'}),
                    IntParameter(name='data_seed', label='Random seed', maximum=999999999, default=42,
                                 commands={'train_model'}),
                    FileParameter(name='hdf5_file', label='', filters=["HDF5 (*.h5 *.hdf5)"], require_existing=True,
                                  default=self.update_file, commands={'train_model'}),
                    CheckParameter(name='time_first', label='Time first', default=False, commands={'train_model'}),
                    CheckParameter(name='augmentation', label='Augmentation', default=False, exclusive=False,
                                   commands={'train_model'}),
                 ]
                )
        set_connections({PyDetecDiv.app.project_selected: [self.update_file]})

    def update_file(self):
        """
        Update the HDF5 file path according to the current project. The default path corresponds to the default path for the HDF5
        ROI creator tool
        """
        if 'cnrs.plewniak.roiseqhdf5creator' in PyDetecDiv.tools:
            self.parameters.hdf5_file.set_value(os.path.join(PyDetecDiv.tools['cnrs.plewniak.roiseqhdf5creator'].working_dir,
                                                             'roi_data.h5'))
