"""
Video classifier tool
"""
import datetime
import locale

import tables
from torch import optim

from pydetecdiv.app.parameters import Parameters, IntParameter, FloatParameter, ChoiceParameter, PathParameter, CheckParameter
from pydetecdiv.app.tools.deep_learning import DeepTool, ModelTrainer, ModelEvaluator, Predictor
from pydetecdiv.domain.tools.video_classifier.train import VideoClassifierTrainer
from pydetecdiv.domain.tools.video_classifier.evaluate import VideoClassifierEvaluator
from pydetecdiv.domain.tools.video_classifier.predict import VideoClassifierPredictor
from pydetecdiv.domain.tools.data.hdf5 import ROIHDF5reader


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
                                   minimum=0.01, maximum=0.98, ),
                    FloatParameter(name='num_validation', label='Validation dataset', groups={'training', 'finetune'},
                                   default=0.3, minimum=0.01, maximum=0.98, ),
                    FloatParameter(name='num_test', label='Test dataset', groups={'training', 'finetune'}, default=0.3,
                                   minimum=0.01, maximum=0.98, enabled=False),
                    IntParameter(name='data_seed', label='Random seed', groups={'training', 'finetune'}, maximum=999999999,
                                 default=42),
                    PathParameter(name='hdf5_file', label='', select_dir=False, groups={'training', 'finetune', 'predict'},
                                  filters=["HDF5 (*.h5 *.hdf5)",], default='roi_data.h5',),
                    CheckParameter(name='time_first', label='Time first', groups={'training', 'finetune', 'predict'},
                                   default=False),
                    ]
                )

    def prepare_data_for_training(self, *args, **kwargs) -> None:
        """
        Prepare the data for training
        """
        print('Preparing data for training')
        print(self.parameters.hdf5_file.value)
        # h5file = tables.open_file(self.parameters.hdf5_file.value, mode='r')
        # hdf5_reader = ROIHDF5reader(h5file)
        hdf5_reader = ROIHDF5reader(tables.open_file(self.parameters.hdf5_file.value, mode='r'),
                                    time_first=self.parameters.time_first.value)
        print(f'{hdf5_reader.roi_data(slice(0, 4), 0).shape}')
        print(f'{hdf5_reader.target(slice(0, 3), slice(0, 5))}')
        print(f'{hdf5_reader.target(0, 0)}')
        # print(f'{hdf5_reader.class_names()[hdf5_reader.target(0, 0)]}')
        print(f'{hdf5_reader.class_names()}')
        print(f'{hdf5_reader.roi_id(0)}')
        print(f'{hdf5_reader.roi(0)}')
        # targets_arr = h5file.root.targets
        # num_frames = targets_arr.shape[0]
        # num_rois = targets_arr.shape[1]
        #
        # print(f'{num_rois} ROIs and {num_frames} frames')
        # class_names = [c[0].decode(locale.getpreferredencoding()) for c in h5file.root.class_names.read()]
        # print(f'Class names: {class_names}')
        # h5file.close()
        hdf5_reader.source.close()
        run = self.save_run(command='prepare_data', param_list=[self.parameters.hdf5_file,
                                                                self.parameters.time_first])
        print(run)

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

    # def save_run(self, *args, **kwargs):
    #     """
    #     Concrete method saving the video classifier run
    #     """
