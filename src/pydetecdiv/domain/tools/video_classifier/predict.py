"""
Video classifier predictor class
"""
from typing import TYPE_CHECKING

from pydetecdiv.app.tools.deep_learning import Predictor

if TYPE_CHECKING:
    from pydetecdiv.domain.tools.video_classifier import VideoClassifier

class VideoClassifierPredictor(Predictor):
    """
    Video classifier predictor class to predict video classes using deep learning video classifier model
    """
    def __init__(self, tool: 'VideoClassifier'):
        super().__init__(tool)

    def predict(self):
        """
        The prediction procedure of the video classifier model
        """
