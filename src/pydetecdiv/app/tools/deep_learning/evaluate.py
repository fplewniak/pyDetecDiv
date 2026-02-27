"""
Abstract ModelEvaluator class
"""
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydetecdiv.app.tools.deep_learning import DeepTool


class ModelEvaluator(ABC):
    """
    Abstract ModelEvaluator class providing the basic functionalities for deep learning model evaluation
    """
    def __init__(self, tool: 'DeepTool'):
        self.tool = tool
        self.tool.command = 'evaluate_model'

    @abstractmethod
    def evaluate_model(self):
        """
        Abstract method to evaluate the model
        """
