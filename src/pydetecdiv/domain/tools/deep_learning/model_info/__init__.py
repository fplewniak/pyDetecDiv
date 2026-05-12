from pydetecdiv.app.tools import Tool

from pydetecdiv.app.parameters import Parameters, StringParameter, ChoiceParameter, IntParameter
from pydetecdiv.domain.tools.video_classifier.models import MViT, Swin3D, S3D, VideoResNet


class ModelInfo(Tool):
    id_ = 'cnrs.plewniak.deeplearningmodelinfo'
    version = '1.0.0'
    name = 'Model information'

    def __init__(self, parameters: Parameters | None = None, working_dir: str | None = None):
        super().__init__(parameters=parameters, working_dir=working_dir)

        self.parameters = Parameters(
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
                        }, label='Model'),
                    IntParameter('num_classes', label='Number of classes', default=6),
                    IntParameter('batch_size', label='Batch size', default=8)
                    ]
                )
