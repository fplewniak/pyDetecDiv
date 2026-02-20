"""
Abstract DeepTool class
"""
from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from pydetecdiv.app.parameters import Parameters
from pydetecdiv.app.tools import Tool

if TYPE_CHECKING:
    from pydetecdiv.app.tools.deep_learning.train import ModelTrainer
    from pydetecdiv.app.tools.deep_learning.evaluate import ModelEvaluator
    from pydetecdiv.app.tools.deep_learning.predict import Predictor


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

    def set_model(self, model: torch.nn.Module):
        """
        Set the deep-learning model used by the tool

        :param model: the deep learning model
        """
        self.model = model
        self.model.to(self.device)

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
    def prepare_data_for_training(self, *args, **kwargs) -> None:
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

    def run_test(self):
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
