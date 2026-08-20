import os
from typing import cast

import pandas as pd
import cv2 as cv

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.core.widgets import set_connections
from pydetecdiv.app.parameters import Parameters, ChoiceParameter
from pydetecdiv.app.tools import Tool, Commands, Command
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.domain.Image import ImgDType


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
                    ChoiceParameter(name='method', label='Method', default='optical flow',
                                    items={'optical flow': None}
                                    # items={'vidstab': None, 'phase correlation cv2': None, 'phase correlation skimage': None, 'optical flow': None}
                                    )
                    ])

        set_connections({PyDetecDiv.app.project_selected: [self.update_fov_list, ]})

    def run_drift_computation(self):
        """
        Compute the drift for the select FOVs
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            fov_list = [cast(FOV, project.get_named_object('FOV', name))
                        for name in self.parameters.FOVs.qmodel.selected_keys()]
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
        """
        Compute drift for the selected FOVs

        :param fov_list: the list of FOVs to compute drift for
        """
        self.count = 0
        for fov in fov_list:
            match self.parameters.method.value:
                case 'optical flow':
                    for i in self.compute_drift_optical_flow(fov):
                        yield i
                # case 'phase correlation cv2':
                #     for i in self.compute_drift_phase_correlation_cv2(fov):
                #         yield i
                # case 'phase correlation skimage':
                #     for i in self.compute_drift_phase_cross_correlation(fov):
                #         yield i
                # case 'vidstab':
                #     for i in self.compute_drift_vidstab(fov):
                #         yield i
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

    def compute_drift_optical_flow(self, fov: FOV, Z: int = 0, C: int = 0):
        df = pd.DataFrame(columns=['dx', 'dy'])
        for frame in range(1, fov.sizeT):
            prev = fov.image(T=frame - 1, Z=Z, C=C, imgdtype=ImgDType.uint8)
            curr = fov.image(T=frame, Z=Z, C=C, imgdtype=ImgDType.uint8)
            points0 = cv.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3,)

            # if points0 is None or len(points0) < 10:
            #     raise RuntimeError("Insufficient features.")

            points1, status, errors = cv.calcOpticalFlowPyrLK(prev, curr, points0, None,)

            good = status.ravel().astype(bool)

            p0 = points0[good].reshape(-1, 2)
            p1 = points1[good].reshape(-1, 2)

            # if len(p0) < 10:
            #     raise RuntimeError("Too few valid optical-flow tracks.")

            # Estimate a 2-D affine transformation.
            #
            # Since we know the true transformation is translation only,
            # we only retain the translation component.
            matrix, inliers = cv.estimateAffinePartial2D( p0, p1,
                    method=cv.RANSAC,
                    ransacReprojThreshold=1.5,
                    maxIters=1000,
                    confidence=0.99,
                    refineIters=10,
                    )

            if matrix is None:
                raise RuntimeError("RANSAC failed.")

            dx = matrix[0, 2]
            dy = matrix[1, 2]
            df.loc[len(df)] = (dx, dy)
            # inlier_fraction = np.mean(inliers)
            self.count += 1
            yield self.count

        self.drift[fov.name] = pd.concat([pd.DataFrame([[0, 0]], columns=['dx', 'dy']), df.cumsum(axis=0)], ignore_index=True)

    def apply_drift_correction(self, tool: Tool):
        """
        Toggle the apply_drift global flag: True if correction should be applied, False otherwise
        """
        PyDetecDiv.app.set_apply_drift(not PyDetecDiv.apply_drift)
