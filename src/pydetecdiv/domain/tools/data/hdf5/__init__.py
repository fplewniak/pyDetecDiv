from pydetecdiv.app.parameters import Parameters, PathParameter
from pydetecdiv.app.tools import Tool


class ROI_HDF5creator(Tool):
    id_ = 'cnrs.plewniak.roihdf5creator'
    version = '1.0.0'
    name = 'ROI HDF5 creator'

    def __init__(self, parameters: Parameters | None = None, working_dir: str | None = None):
        super().__init__(parameters, working_dir)
        self.parameters = Parameters(
                [
                    PathParameter(name='destination_dir', label='Destination', select_dir=True),
                    ]
                )

    def save_run(self, *args, **kwargs):
        pass

