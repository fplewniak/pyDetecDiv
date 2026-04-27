from pydetecdiv.app.tools import Tool

from pydetecdiv.app.parameters import Parameters, StringParameter


class ClassificationSchemeManagement(Tool):
    id_ = 'cnrs.plewniak.classificationschemes'
    version = '1.0.0'
    name = 'Classification schemes'

    def __init__(self, parameters: Parameters | None = None, working_dir: str | None = None):
        super().__init__(parameters=parameters, working_dir=working_dir)

        self.parameters = Parameters(
                [
                    StringParameter('name', label='Name'),
                    # ('num_classes', label='Number of classes', default=6),
                    ]
                )
