"""
Tools providing information about deep learning models
"""
from pydetecdiv.app.tools import Tool

from pydetecdiv.app.parameters import Parameters, StringParameter, ChoiceParameter, IntParameter, Parameter
from pydetecdiv.domain.tools.video_classifier.models import MViT, Swin3D, S3D, VideoResNet


class ModelInfo(Tool):
    """
    Tool to display a deep learning model summary
    """
    id_ = 'cnrs.plewniak.deeplearningmodelinfo'
    version = '1.0.0'
    name = 'Model information'

    def __init__(self, parameters: Parameters = Parameters(), working_dir: str | None = None):
        super().__init__(parameters=parameters, working_dir=working_dir)

        self.parameters.update_parameters(
                [
                    ChoiceParameter('model', items={
                        'MViT_v2_small': MViT.MViT_v2_s,
                        'MViT_v1_b': MViT.MViT_v1_b,
                        'Swin3D_tiny': Swin3D.Swin3D_tiny,
                        'Swin3D_small': Swin3D.Swin3D_small,
                        'Swin3D_base': Swin3D.Swin3D_base,
                        'S3D': S3D.S3D,
                        'R3D_18': VideoResNet.R3D_18,
                        'MC3_18': VideoResNet.MC3_18,
                        'R2+1d_18': VideoResNet.R2Plus1d_18,
                        'CustomR2Plus_1D': VideoResNet.CustomR2Plus_1D,
                        }, label='Model', default='R2+1d_18'),
                    IntParameter('num_classes', label='Number of classes', default=6),
                    StringParameter(name='layers', label='Blocks layers', default='[1, 2]'),
                    StringParameter(name='strides', label='Strides', default='[1, 2]'),
                    IntParameter('batch_size', label='Batch size', default=8),
                    IntParameter('depth', label='Depth', default=3)
                    ]
                )

    def save_run(self, command: str | None = None, param_list: list[Parameter] | None = None, key_val: dict | None = None) -> None:
        pass
