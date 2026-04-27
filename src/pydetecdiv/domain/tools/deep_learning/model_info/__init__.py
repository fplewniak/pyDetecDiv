from pydetecdiv.app.tools import Tool

from pydetecdiv.app.parameters import Parameters, StringParameter, ChoiceParameter, IntParameter
from pydetecdiv.domain.tools.video_classifier.models.MViT import MViT_v2_s, MViT_v1_b


class ModelInfo(Tool):
    id_ = 'cnrs.plewniak.deeplearningmodelinfo'
    version = '1.0.0'
    name = 'Model information'

    def __init__(self, parameters: Parameters | None = None, working_dir: str | None = None):
        super().__init__(parameters=parameters, working_dir=working_dir)

        self.parameters = Parameters(
                [
                    ChoiceParameter('model', items={'MViT_v2_small': MViT_v2_s, 'MViT_v1_b': MViT_v1_b, }, label='Model'),
                    IntParameter('num_classes', label='Number of classes', default=6),
                    IntParameter('batch_size', label='Batch size', default=8)
                    ]
                )
