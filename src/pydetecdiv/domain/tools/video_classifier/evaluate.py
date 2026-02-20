"""
Video classifier evaluator class
"""
from pydetecdiv.app.tools.deep_learning import DeepTool, ModelEvaluator


class VideoClassifierEvaluator(ModelEvaluator):
    """
    Video classifier evaluator class to run the evaluation of deep learning video classifier model
    """
    def __init__(self, tool: DeepTool):
        super().__init__(tool)

    def evaluate_model(self):
        """
        The evaluation procedure of the video classifier model
        """
