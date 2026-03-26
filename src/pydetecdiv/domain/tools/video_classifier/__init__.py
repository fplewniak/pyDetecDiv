"""
Video classifier tool
"""
from torch import optim

from pydetecdiv.app.parameters import Parameters, IntParameter, FloatParameter, ChoiceParameter
from pydetecdiv.app.tools.deep_learning import DeepTool, ModelTrainer, ModelEvaluator, Predictor
from pydetecdiv.domain.tools.video_classifier.train import VideoClassifierTrainer
from pydetecdiv.domain.tools.video_classifier.evaluate import VideoClassifierEvaluator
from pydetecdiv.domain.tools.video_classifier.predict import VideoClassifierPredictor


class VideoClassifier(DeepTool):
    """
    Video classifier tool, providing all functionalities for deep learning video classification (training, evaluation, prediction)
    """
    id_ = 'cnrs.plewniak.videoclassifier'
    version = '1.0.0'
    name = 'Video Classifier'

    def __init__(self, parameters: Parameters | None = None, working_dir: str | None = None):
        super().__init__(parameters, working_dir)
        self.parameters = Parameters(
                [
                    IntParameter(name='epochs', label='Epochs', groups={'training', 'finetune'}, default=32),
                    IntParameter(name='batch_size', label='Batch size', groups={'training', 'finetune'}, default=32, ),
                    ChoiceParameter(name='optimizer', label='Optimizer', groups={'training', 'finetune'}, default='AdamW',
                                    items={'AdamW'   : optim.AdamW,
                                           'SGD'     : optim.SGD,
                                           'Adadelta': optim.Adadelta,
                                           'Adamax'  : optim.Adamax,
                                           'Nadam'   : optim.NAdam,
                                           },),
                    IntParameter(name='seed', label='Random seed', groups={'training', 'finetune'}, maximum=999999999,
                                 default=42, ),

                    FloatParameter(name='num_training', label='Training dataset', groups={'training', 'finetune'}, default=0.4,
                                   minimum=0.01, maximum=0.99, ),
                    FloatParameter(name='num_validation', label='Validation dataset', groups={'training', 'finetune'},
                                   default=0.3, minimum=0.01, maximum=0.99, ),
                    FloatParameter(name='num_test', label='Test dataset', groups={'training', 'finetune'}, default=0.3,
                                   minimum=0.01, maximum=0.99, enabled=False),
                    IntParameter(name='data_seed', label='Random seed', groups={'training', 'finetune'}, maximum=999999999,
                                 default=42),
                    ]
                )

    def prepare_data_for_training(self, *args, **kwargs) -> None:
        """
        Prepare the data for training
        """

    def prepare_data_for_prediction(self, *args, **kwargs) -> None:
        """
        Prepare the data for prediction
        """

    def create_trainer(self) -> VideoClassifierTrainer:
        """
        Concrete factory to create video classifier trainer
        """
        return VideoClassifierTrainer(self)

    def create_evaluator(self) -> VideoClassifierEvaluator:
        """
        Concrete factory to create video classifier evaluator
        """
        return VideoClassifierEvaluator(self)

    def create_predictor(self) -> VideoClassifierPredictor:
        """
        Concrete factory to create video classifier predictor
        """
        return VideoClassifierPredictor(self)

    def save_run(self, *args, **kwargs):
        """
        Concrete method saving the video classifier run
        """
