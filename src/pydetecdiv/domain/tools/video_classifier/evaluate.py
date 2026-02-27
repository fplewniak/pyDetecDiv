"""
Video classifier evaluator class
"""
from typing import TYPE_CHECKING

from pydetecdiv.app.tools.deep_learning import ModelEvaluator

if TYPE_CHECKING:
    from pydetecdiv.domain.tools.video_classifier import VideoClassifier

class VideoClassifierEvaluator(ModelEvaluator):
    """
    Video classifier evaluator class to run the evaluation of deep learning video classifier model
    """
    def __init__(self, tool: 'VideoClassifier'):
        super().__init__(tool)

    def evaluate_model(self):
        """
        The evaluation procedure of the video classifier model
        """
