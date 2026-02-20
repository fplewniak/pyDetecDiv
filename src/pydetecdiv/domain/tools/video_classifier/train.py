"""
Video classifier trainer class
"""
from pydetecdiv.app.tools.deep_learning import ModelTrainer, DeepTool


class VideoClassifierTrainer(ModelTrainer):
    """
    Video classifier trainer class to run the training of deep learning video classifier model
    """
    def __init__(self, tool: DeepTool):
        super().__init__(tool)

    def train_model(self):
        """
        Train the video classifier model, running the training loop once per epoch for as many epochs as requested by the user
        """

    def training_loop(self):
        """
        The training loop for the video classifier, run once per epoch
        """
