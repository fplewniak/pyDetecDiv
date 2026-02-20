"""
Abstract ModelTrainer class
"""
from abc import ABC, abstractmethod

from pydetecdiv.app.tools.deep_learning import DeepTool


class ModelTrainer(ABC):
    """
    Abstract ModelTrainer class providing the basic functionalities for model training of deep learning tools
    """
    def __init__(self, tool: DeepTool):
        self.tool = tool
        self.tool.command = 'train_model'

    @abstractmethod
    def train_model(self):
        """
        The global training procedure. This method should be implemented by subclasses to run the training loop for as many epochs
        as requested by the user.
        """

    @abstractmethod
    def training_loop(self):
        """
        The elementary training loop, run once per epoch on all batches
        """
