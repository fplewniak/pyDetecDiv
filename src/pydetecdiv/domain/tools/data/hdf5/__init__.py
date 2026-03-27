import time
import torch

import fastremap
import numpy as np
import tables as tbl

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
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
        if self.parameters.annotations:
            print(f'Create ROI HDF5 file with annotations: {self.parameters.hdf5_file}')
        else:
            print(f'Create ROI HDF5 file: {self.parameters.hdf5_file}')
        seqlen = self.parameters.seqlen.value
        h5file = tbl.open_file(self.parameters.hdf5_file.value, mode='w', title='ROI data')

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            num_rois = project.count_objects('ROI')
            num_frames = np.max([fov.image_resource().sizeT for fov in project.get_objects('FOV')])
            height = np.int64(np.max([roi.height for roi in project.get_objects('ROI')]) + 1)
            width = np.int64(np.max([roi.width for roi in project.get_objects('ROI')]) + 1)
            print(num_frames, height, width)
            roi_seq = h5file.create_carray(h5file.root, 'roi_seq', atom=tbl.Float16Atom(shape=(seqlen, height, width, np.int64(3))),
                                           chunkshape=(num_frames, 1,), shape=(num_frames, num_rois))
            roi_ids = h5file.create_carray(h5file.root,  'roi_ids', atom=tbl.UInt16Atom(shape=(1,)),
                                           chunkshape=(num_rois,), shape=(num_rois,))
            roi_id_values = np.array(sorted([roi.id_ for roi in project.get_objects('ROI')]))
            roi_new_idx, roi_mapping = fastremap.renumber(roi_id_values, in_place=False, preserve_zero=False)
            for idx in roi_new_idx:
                roi_ids[idx - 1] = roi_id_values[idx - 1]

            # start = time.perf_counter()
            # for fov in project.get_objects('FOV'):
            #     start_fov = time.perf_counter()
            #     print(fov)
            #     image_resource_data = fov.image_resource().image_resource_data()
            #     # print(image_resource_data.dask_array.chunksize)
            #     # print(image_resource_data.dask_array.chunks)
            #     for roi in fov.roi_list:
            #         start_partiel = time.perf_counter()
            #         (x1, y1), (x2, y2) = (roi.top_left, roi.bottom_right)
            #         t = 0
            #         seq = image_resource_data.sequence(seqlen, T=0, crop=(slice(x1, x2+1), slice(y1, y2+1)), drift=True)
            #
            #         for t in range(1, image_resource_data.sizeT - seqlen, 1):
            #             # seq = image_resource_data.sequence(seqlen, T=t, crop=(slice(x1, x2+1), slice(y1, y2+1)), drift=True)
            #             img = image_resource_data.auto_channels(T=t, crop=(slice(x1, x2+1), slice(y1, y2+1)), drift=True)
            #             seq = torch.cat([seq[1:], img.as_tensor().unsqueeze(dim=0)], dim=0)
            #         print(f'{roi.name}: {time.perf_counter() - start_partiel} s')
            #     print(f'{fov.name}: {time.perf_counter() - start_fov}')

            h5file.close()
            # print(f'Full job in {time.perf_counter() - start_fov} s')

    def save_run(self, *args, **kwargs):
        pass

