import os
from typing import cast

import numpy as np
import pandas as pd
import cv2 as cv
from vidstab import VidStab

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.core.widgets import set_connections
from pydetecdiv.app.parameters import Parameters, ChoiceParameter
from pydetecdiv.app.tools import Tool, Commands, Command
from pydetecdiv.domain.FOV import FOV


class DriftCorrection(Tool):
    """
    Class defining the tool for drift correction
    """
    id_ = 'cnrs.plewniak.driftcorrection'
    version = '1.0.0'
    name = 'Drift correction'

    def __init__(self, parameters: Parameters = Parameters(), commands: Commands = Commands(), working_dir: str | None = None):
        super().__init__(parameters, commands, working_dir)

        self.drift = {}
        self.count = 0

        self.commands.update([
            Command('compute_drift', 'Compute drift', self.run_drift_computation),
            Command('apply_drift_correction', 'Apply drift correction', self.apply_drift_correction)
            ])

        self.parameters.update_parameters(
                commands={'compute_drift'},
                parameters=[
                    ChoiceParameter(name='FOVs', label='FOV', updater=self.update_fov_list, multiselection=True),
                    ChoiceParameter(name='method', label='Method', default='vidstab',
                                    items={'vidstab': None, 'phase correlation': None})
                    ])

        set_connections({PyDetecDiv.app.project_selected: [self.update_fov_list, ]})

    def run_drift_computation(self):
        """
        Compute the drift for the select FOVs
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            fov_list = [project.get_named_object('FOV', name) for name in self.parameters.FOVs.qmodel.selected_keys()]
            total = sum([fov.sizeT for fov in fov_list])
            for i in self.compute_drift(fov_list):
                yield 100.0 * float(i) / float(total)
        self.save_run()

    def update_fov_list(self) -> None:
        """
        Return the list of FOVs in the project as a dictionary mapping actual FOV objects to their names

        :return: a dictionary of FOVs in project
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.parameters.FOVs.set_items({cast(FOV, fov).name: cast(FOV, fov) for fov in project.get_objects('FOV')})

    def compute_drift(self, fov_list: list[FOV]):
        self.count = 0
        for fov in fov_list:
            match self.parameters.method.value:
                case 'phase correlation':
                    for i in self.compute_drift_phase_correlation_cv2(fov):
                        yield i
                case 'vidstab':
                    for i in self.compute_drift_vidstab(fov):
                        yield i
            image_resource = fov.image_resource()
            if image_resource.key_val is None:
                image_resource.key_val = {}
            drift_file = f'{fov.name}_drift_data.csv'
            drift_path = os.path.join(self.working_dir, drift_file)
            self.drift[fov.name].to_csv(drift_path, float_format='%.3f', columns=['dx', 'dy'], index=False)
            fov.set_timestamp()
            image_resource.key_val.update({'drift': drift_file, 'drift method': self.parameters.method.value})
            image_resource.validate()
            image_resource.project.commit()

    def compute_drift_phase_correlation_cv2(self, fov: FOV, Z: int = 0, C: int = 0):
        """
        Compute the cumulative transforms (dx, dy) to apply in order to correct the drift using phase correlation

        :param Z: the layer index
        :type Z: int
        :param C: the channel index
        :type C: int
        :param max_mem: maximum memory use when using memory mapped TIFF
        :type max_mem: int
        :return: the cumulative drift transforms dx, dy, dr
        :rtype: pandas DataFrame
        """
        df = pd.DataFrame(columns=['dx', 'dy'])
        for frame in range(1, fov.sizeT):
            df.loc[len(df)], _ = cv.phaseCorrelate(np.float32(fov.image(T=frame - 1, Z=Z, C=C)),
                                                   np.float32(fov.image(T=frame, Z=Z, C=C)))
            self.count += 1
            yield self.count
        df.cumsum(axis=0)
        self.drift[fov.name] = pd.concat([pd.DataFrame([[0, 0]], columns=['dx', 'dy']), df], ignore_index=True)

    def compute_drift_vidstab(self, fov: FOV, Z: int = 0, C: int = 0, smoothing_window: int = 1):
        """
        Compute the cumulative transforms (dx, dy, dr) to apply in order to stabilize the time series and correct drift

        :param Z: the layer index
        :type Z: int
        :param C: the channel index
        :type C: int
        :param max_mem: maximum memory use when using memory mapped TIFF
        :type max_mem: int
        :return: the cumulative drift transforms dx, dy, dr
        :rtype: pandas DataFrame
        """
        stabilizer = VidStab()
        for frame in range(0, fov.sizeT):
            _ = stabilizer.stabilize_frame(
                    input_frame=np.uint8(np.array(fov.image(T=frame, Z=Z, C=C)) / 65535 * 255), smoothing_window=smoothing_window)
            self.count += 1
            yield self.count
        df = pd.DataFrame(stabilizer.transforms, columns=('dx', 'dy', 'dr')).cumsum(axis=0)[['dx', 'dy']]
        self.drift[fov.name] = pd.concat([pd.DataFrame([[0, 0]], columns=['dx', 'dy']), df], ignore_index=True)

    def apply_drift_correction(self, tool):
        PyDetecDiv.app.set_apply_drift(not PyDetecDiv.apply_drift)
        print(PyDetecDiv.apply_drift)
