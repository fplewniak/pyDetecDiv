"""
Abstract DeepTool class
"""
from abc import abstractmethod

import polars
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2, transforms, functional as F

from pydetecdiv.app.parameters import Parameters
from pydetecdiv.app.tools import Tool

from pydetecdiv.app.tools.deep_learning.train import ModelTrainer
from pydetecdiv.app.tools.deep_learning.evaluate import ModelEvaluator
from pydetecdiv.app.tools.deep_learning.predict import Predictor
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.tools.data import RoiDataReader


class ROIDataset(Dataset):
    def __init__(self, data_reader: RoiDataReader, indices: polars.DataFrame, targets: bool = False,
                 image_shape: tuple[int, int] = (60, 60), transform: torch.nn.Module = None):
        self.reader = data_reader
        self.indices = indices
        self.targets = targets
        self.image_shape = list(image_shape)
        self.transform = v2.ToDtype(torch.float, scale=True)
        if transform:
            self.transform = transforms.Compose([self.transform, transform])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor, Tensor]:
        df = self.indices[idx]
        roi_idx = df['roi'].item()
        frame_idx = df['frame'].item()
        item = self.reader.roi_data(roi_idx=roi_idx, frame=frame_idx)
        item = F.resize(item, size=self.image_shape)
        if self.transform:
                item = self.transform(item)
        if self.targets:
            target = self.reader.target(roi_idx=roi_idx, frame=frame_idx)
            return item, target
        return item

    def close(self):
        self.reader.close()

    @property
    def class_names(self):
        return self.reader.class_names

    def roi(self, idx: int) -> ROI:
        return self.reader.roi(idx)

    def get_ref(self, idx: int) -> tuple[int, int]:
        df = self.indices[idx]
        roi_idx = df['roi'].item()
        frame_idx = df['frame'].item()
        return self.reader.roi_id(roi_idx), frame_idx


class DeepTool(Tool):
    """
    DeepTool abstract class providing the basic functionality for deep-learning tools
    """

    def __init__(self, parameters: Parameters | None = None, working_dir: str | None = None, device: torch.device | None = None,
                 model: torch.nn.Module = None):
        super().__init__(parameters=parameters, working_dir=working_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = model
        if self.model is not None:
            self.model.to(self.device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = None, None, None
        self.dataloader = None
        self._model_trainer = None
        self._model_evaluator = None
        self._model_predictor = None

    def set_model(self, model: torch.nn.Module):
        """
        Set the deep-learning model used by the tool

        :param model: the deep learning model
        """
        self.model = model
        self.model.to(self.device)

    @property
    def model_trainer(self) -> 'ModelTrainer':
        if self._model_trainer is None:
            self._model_trainer = self.create_trainer()
        return self._model_trainer

    @property
    def model_evaluator(self) -> 'ModelEvaluator':
        if self._model_evaluator is None:
            self._model_evaluator = self.create_evaluator()
        return self._model_evaluator

    @property
    def model_predictor(self) -> 'Predictor':
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
    def prepare_data_for_training(self, *args, **kwargs) -> tuple[ROIDataset, ROIDataset]:
        """
        Abstract method to prepare the data for training
        """

    @abstractmethod
    def prepare_data_for_prediction(self, *args, **kwargs) -> None:
        """
        Abstract method to prepare the data for prediction
        """

    def run_training(self):
        """
        Generic method calling the trainer factory and training the model on the training and validation datasets
        """
        model_trainer = self.create_trainer()
        model_trainer.train_model()

    def run_evaluation(self):
        """
        Generic method calling the evaluator factory and evaluating the model on the test dataset
        """
        model_evaluator = self.create_evaluator()
        model_evaluator.evaluate_model()

    def predict(self):
        """
        Generic method calling the predictor factory to make prediction with the model
        """
        predictor = self.create_predictor()
        predictor.predict()
