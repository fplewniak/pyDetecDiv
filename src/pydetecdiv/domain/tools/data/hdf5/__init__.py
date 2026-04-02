import time
import torch

import fastremap
import numpy as np
import tables as tbl

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.app.parameters import Parameters, PathParameter, CheckParameter, IntParameter, ChoiceParameter
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
                    ChoiceParameter(name='red_channel', label='Red', default='0', updater=self.update_channels),
                    ChoiceParameter(name='green_channel', label='Green', default='0', updater=self.update_channels),
                    ChoiceParameter(name='blue_channel', label='Blue', default='0', updater=self.update_channels),
                    ]
                )

    def update_channels(self) -> None:
        """
        Updates the list of available channels to display in the GUI form
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            image_resource = project.get_object('ImageResource', 1)
            n_layers = image_resource.zdim if image_resource else 0

        for param in ['red_channel', 'green_channel', 'blue_channel']:
            self.parameters[param].set_items({str(i): i for i in range(n_layers)})

    def create_file(self):
        if self.parameters.annotations:
            print(f'Create ROI HDF5 file with annotations: {self.parameters.hdf5_file}')
        else:
            print(f'Create ROI HDF5 file: {self.parameters.hdf5_file}')
        seqlen = self.parameters.seqlen.value
        h5file = tbl.open_file(self.parameters.hdf5_file.value, mode='w', title='ROI data')
        z_channels = [self.parameters.red_channel.value, self.parameters.green_channel.value, self.parameters.blue_channel.value]

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            num_rois = project.count_objects('ROI')
            num_frames = int(np.max([fov.image_resource().sizeT for fov in project.get_objects('FOV')]))
            num_sequences = num_frames - seqlen
            height = np.int64(np.max([roi.height for roi in project.get_objects('ROI')]))
            width = np.int64(np.max([roi.width for roi in project.get_objects('ROI')]))
            roi_seq_hdf5 = h5file.create_carray(h5file.root, 'roi_seq', atom=tbl.Float16Atom(shape=(seqlen, np.int64(3), height, width)),
                                           chunkshape=(num_frames, 1,), shape=(num_frames, num_rois))
            roi_ids_hdf5 = h5file.create_carray(h5file.root,  'roi_ids', atom=tbl.UInt16Atom(shape=(np.int64(1),)),
                                           chunkshape=(num_rois,), shape=(num_rois,))
            if self.parameters.annotations:
                targets_hdf5 = h5file.create_carray(h5file.root, 'targets', atom=tbl.Float16Atom(shape=(np.int64(1),)),
                                           chunkshape=(num_sequences, 1,), shape=(num_sequences, num_rois))
                roi_id_values = np.array(sorted([roi.id_ for roi in project.get_annotated_rois()]))
            else:
                roi_id_values = np.array(sorted([roi.id_ for roi in project.get_objects('ROI')]))

            roi_new_idx, roi_mapping = fastremap.renumber(roi_id_values, in_place=False, preserve_zero=False)
            for idx in roi_new_idx:
                roi_ids_hdf5[idx - 1] = roi_id_values[idx - 1]

            start = time.perf_counter()
            for fov in project.get_objects('FOV'):
                start_fov = time.perf_counter()
                print(fov)
                image_resource_data = fov.image_resource().image_resource_data()
                # print(image_resource_data.dask_array.chunksize)
                # print(image_resource_data.dask_array.chunks)
                for roi in fov.roi_list:
                    start_partiel = time.perf_counter()
                    (x1, y1), (x2, y2) = (roi.top_left, roi.bottom_right)
                    t = 0
                    roi_seq = image_resource_data.sequence(seqlen, T=0, Z=z_channels, crop=(slice(x1, x2+1), slice(y1, y2+1)),
                                                       drift=True, resize=(height, width))
                    roi_seq_hdf5[t, roi_mapping[roi.id_] - 1] = roi_seq.numpy()

                    for t in range(1, image_resource_data.sizeT - seqlen, 1):
                        # seq = image_resource_data.sequence(seqlen, T=t, crop=(slice(x1, x2+1), slice(y1, y2+1)), drift=True)
                        img = image_resource_data.auto_channels(T=t, Z=z_channels, crop=(slice(x1, x2+1), slice(y1, y2+1)),
                                                                drift=True, resize=(height, width))
                        roi_seq = torch.cat([roi_seq[1:], img.as_tensor().unsqueeze(dim=0)], dim=0)
                        roi_seq_hdf5[t, roi_mapping[roi.id_] - 1] = roi_seq.numpy()
                    print(f'{roi.name}: {time.perf_counter() - start_partiel} s')
                print(f'{fov.name}: {time.perf_counter() - start_fov}')

            h5file.close()
            print(f'Full job in {time.perf_counter() - start} s')

    def save_run(self, *args, **kwargs):
        pass

