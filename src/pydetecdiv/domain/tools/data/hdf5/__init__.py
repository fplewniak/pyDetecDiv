import locale
import time

import tables
import torch

import fastremap
import numpy as np
import tables as tbl

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.app.parameters import Parameters, PathParameter, CheckParameter, IntParameter, ChoiceParameter
from pydetecdiv.app.tools import Tool
from pydetecdiv.domain.tools.data import RoiDataReader
from pydetecdiv.utils import hdf5


class ROIseqHDF5creator(Tool):
    id_ = 'cnrs.plewniak.roiseqhdf5creator'
    version = '1.0.0'
    name = 'ROI HDF5 creator'

    def __init__(self, parameters: Parameters | None = None, working_dir: str | None = None):
        super().__init__(parameters, working_dir)
        self.parameters = Parameters(
                [
                    PathParameter(name='hdf5_file', label='', select_dir=False, filters=["HDF5 (*.h5 *.hdf5)", ],
                                  default='roi_data.h5', ),
                    CheckParameter(name='annotations', label='Annotated ROIs', default=True),
                    ChoiceParameter(name='classification', label='Classes', updater=self.update_classification),
                    IntParameter(name='seqlen', label='Sequence length', default=15),
                    ChoiceParameter(name='red_channel', label='Red', default='0', updater=self.update_channels),
                    ChoiceParameter(name='green_channel', label='Green', default='0', updater=self.update_channels),
                    ChoiceParameter(name='blue_channel', label='Blue', default='0', updater=self.update_channels),
                    CheckParameter(name='time_first', label='Time first', default=False),
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

    def update_classification(self) -> None:
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.parameters.classification.set_items({f'{c.name} {c.classes}': c for c in project.get_objects('Classification')})

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
            if self.parameters.time_first:
                roi_seq_hdf5 = h5file.create_carray(h5file.root, 'roi_data',
                                                    atom=tbl.Float16Atom(shape=(seqlen, np.int64(3), height, width)),
                                                    chunkshape=(num_frames, 1,), shape=(num_frames, num_rois))
            else:
                roi_seq_hdf5 = h5file.create_carray(h5file.root, 'roi_data',
                                                    atom=tbl.Float16Atom(shape=(seqlen, np.int64(3), height, width)),
                                                    chunkshape=(1, num_frames,), shape=(num_rois, num_frames))
            roi_ids_hdf5 = h5file.create_carray(h5file.root, 'roi_ids', atom=tbl.UInt16Atom(shape=()),
                                                chunkshape=(num_rois,), shape=(num_rois,))
            if self.parameters.annotations:
                if self.parameters.time_first:
                    targets_hdf5 = h5file.create_carray(h5file.root, 'targets', atom=tbl.UInt16Atom(shape=()),
                                                        chunkshape=(num_sequences, 1,), shape=(num_sequences, num_rois))
                else:
                    targets_hdf5 = h5file.create_carray(h5file.root, 'targets', atom=tbl.UInt16Atom(shape=()),
                                                        chunkshape=(1, num_sequences,), shape=(num_rois, num_sequences,))
                classes_hdf5 = h5file.create_table(h5file.root, 'class_names', hdf5.TblNamesRow, 'Class names')
                classes_hdf5.append([(name,) for name in self.parameters.classification.value.classes])
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
                    roi_seq = image_resource_data.sequence(seqlen, T=0, Z=z_channels, crop=(slice(x1, x2 + 1), slice(y1, y2 + 1)),
                                                           drift=True, resize=(height, width))

                    if self.parameters.time_first:
                        roi_seq_hdf5[t, roi_mapping[roi.id_] - 1] = roi_seq.numpy()
                    else:
                        roi_seq_hdf5[roi_mapping[roi.id_] - 1, t] = roi_seq.numpy()

                    if self.parameters.annotations:
                        targets = roi.annotations()
                        if self.parameters.time_first:
                            targets_hdf5[t, roi_mapping[roi.id_] - 1] = targets[t + int(seqlen / 2)].annotation
                        else:
                            targets_hdf5[roi_mapping[roi.id_] - 1, t] = targets[t + int(seqlen / 2)].annotation

                    for t in range(1, image_resource_data.sizeT - seqlen, 1):
                        # seq = image_resource_data.sequence(seqlen, T=t, crop=(slice(x1, x2+1), slice(y1, y2+1)), drift=True)
                        img = image_resource_data.auto_channels(T=t, Z=z_channels, crop=(slice(x1, x2 + 1), slice(y1, y2 + 1)),
                                                                drift=True, resize=(height, width))
                        roi_seq = torch.cat([roi_seq[1:], img.as_tensor().unsqueeze(dim=0)], dim=0)

                        if self.parameters.time_first:
                            roi_seq_hdf5[t, roi_mapping[roi.id_] - 1] = roi_seq.numpy()
                            if self.parameters.annotations and t < (len(targets) - int(seqlen / 2)):
                                targets_hdf5[t, roi_mapping[roi.id_] - 1] = targets[t + int(seqlen / 2)].annotation
                        else:
                            roi_seq_hdf5[roi_mapping[roi.id_] - 1, t] = roi_seq.numpy()
                            if self.parameters.annotations and t < (len(targets) - int(seqlen / 2)):
                                targets_hdf5[roi_mapping[roi.id_] - 1, t] = targets[t + int(seqlen / 2)].annotation
                    print(f'{roi.name}: {time.perf_counter() - start_partiel} s')
                print(f'{fov.name}: {time.perf_counter() - start_fov}')

            h5file.close()
            print(f'Full job in {time.perf_counter() - start} s')

    def save_run(self, *args, **kwargs):
        pass


class ROIHDF5reader(RoiDataReader):
    def __init__(self, source: tables.File, time_first: bool = False):
        super().__init__(source)
        self.targets = source.__contains__('/targets')
        self.time_first = time_first

    def roi_data(self, roi_idx: int | slice = None, frame: int | slice = 0) -> torch.Tensor:
        if self.time_first:
            return self.source.root.roi_data[frame, roi_idx]
        return self.source.root.roi_data[roi_idx, frame]

    def target(self, roi_idx: int | slice = None, frame: int | slice = 0) -> torch.Tensor | None:
        if self.targets:
            if self.time_first:
                return self.source.root.targets[frame, roi_idx]
            return self.source.root.targets[roi_idx, frame]
        return None

    def class_names(self) -> list[str] | None:
        if self.targets:
            return [c[0].decode(locale.getpreferredencoding()) for c in self.source.root.class_names.read()]
        return None

    def roi_id(self, roi_idx: int = None) -> int:
        return self.source.root.roi_ids[roi_idx]

    @property
    def roi_ids(self):
        return self.source.root.roi_ids[:]

    def num_rois(self) -> int:
        return len(self.source.root.roi_ids)

    def num_targets(self) -> int:
        if self.targets:
            if self.time_first:
                return len(self.source.root.targets)
            return int(self.source.root.targets.shape[-1])
        return 0

