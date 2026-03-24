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
                    PathParameter(name='hdf5_file', label='HDF5 file', select_dir=False, filters=["HDF5 (*.h5 *.hdf5)",],
                                  default='roi_data.h5'),
                    ]
                )

    def save_run(self, *args, **kwargs):
        pass

