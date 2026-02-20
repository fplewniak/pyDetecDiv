"""
Video classifier predictor class
"""
from pydetecdiv.app.tools.deep_learning import DeepTool, Predictor


class VideoClassifierPredictor(Predictor):
    """
    Video classifier predictor class to predict video classes using deep learning video classifier model
    """
    def __init__(self, tool: DeepTool):
        super().__init__(tool)

    def predict(self):
        """
        The prediction procedure of the video classifier model
        """
