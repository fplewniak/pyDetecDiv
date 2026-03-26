from pydetecdiv.app.parameters import Parameters, PathParameter, CheckParameter, IntParameter
from pydetecdiv.app.tools import Tool


class ROI_HDF5creator(Tool):
    id_ = 'cnrs.plewniak.roihdf5creator'
    version = '1.0.0'
    name = 'ROI HDF5 creator'

    def __init__(self, parameters: Parameters | None = None, working_dir: str | None = None):
        super().__init__(parameters, working_dir)
        self.parameters = Parameters(
                [
                    PathParameter(name='hdf5_file', label='', select_dir=False, filters=["HDF5 (*.h5 *.hdf5)",],
                                  default='roi_data.h5',),
                    CheckParameter(name='annotations', label='Annotated ROIs', default=True),
                    IntParameter(name='seqlen', label='Sequence length', default=15),
                    ]
                )

    def create_file(self):
        if self.parameters.annotations.value:
            print(f'Create ROI HDF5 file with annotations: {self.parameters.hdf5_file.value}')
        else:
            print(f'Create ROI HDF5 file: {self.parameters.hdf5_file.value}')
        print(f'Sequence length: {self.parameters.seqlen.value}')

        print(self.parameters.hdf5_file > 'ABC')
        print(self.parameters.seqlen > 2, self.parameters.seqlen < 3)

    def save_run(self, *args, **kwargs):
        pass

