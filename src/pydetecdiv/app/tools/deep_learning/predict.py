"""
Abstract Predictor class
"""
from abc import abstractmethod, ABC

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydetecdiv.app.tools.deep_learning import DeepTool


class Predictor(ABC):
    """
    Abstract Predictor class providing the basic functionalities for deep learning prediction
    """
    def __init__(self, tool: 'DeepTool'):
        self.tool = tool
        self.tool.command = 'predict'

    @abstractmethod
    def predict(self):
        """
        Abstract method for prediction
        """
