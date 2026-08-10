import os
from typing import cast

import numpy as np
import pandas as pd
from vidstab import VidStab
import cv2 as cv

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.core.widgets import set_connections
from pydetecdiv.app.parameters import Parameters, ChoiceParameter
from pydetecdiv.app.tools import Tool, Commands, Command
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.settings import get_config_value


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

        self.commands.update([
            Command('compute_drift', 'Compute drift', self.run_drift_computation)
            ])

        self.parameters.update_parameters(
                commands={'compute_drift'},
                parameters=[
                    ChoiceParameter(name='FOVs', label='FOV', updater=self.update_fov_list, multiselection=True),
                    ChoiceParameter(name='method', label='Method', default='vidstab',
                                    items={'vidstab': None, 'phase correlation': None})
                    ])

        set_connections({PyDetecDiv.app.project_selected: [self.update_fov_list,]})

    def run_drift_computation(self):
        """
        Compute the drift for the select FOVs
        """
        fov_list = self.parameters.FOVs.qmodel.selected_values()
        total = sum([fov.sizeT for fov in fov_list])
        for i, fov in enumerate(fov_list):
            self.drift[fov.name] = fov.image_resource().image_resource_data().compute_drift(method=self.parameters.method.value)
            yield 100.0 * float(i) / float(len(fov_list))


    def update_fov_list(self) -> None:
        """
        Return the list of FOVs in the project as a dictionary mapping actual FOV objects to their names

        :return: a dictionary of FOVs in project
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.parameters.FOVs.set_items({cast(FOV, fov).name: cast(FOV, fov) for fov in project.get_objects('FOV')})
