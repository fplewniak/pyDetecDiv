"""
GUI classes for video classifier model training
"""
from typing import Any

from PySide6.QtWidgets import QMenu

from pydetecdiv.app.tools import Tool

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.core.widgets import set_connections
from pydetecdiv.app.gui.tools import ToolAction, ToolDialog
from pydetecdiv.app.gui.tools.deep_learning import plot_training_results
from pydetecdiv.app.parameters import FloatParameter
from pydetecdiv.app.tools.deep_learning import DeepTool
from pydetecdiv.persistence.project import project_exists



class TrainModelDialog(ToolDialog):
    """
    Dialog to choose the parameters for training a Video classifier model
    """

    def __init__(self, tool: DeepTool, **kwargs):
        super().__init__(tool, title='Training Video classifier', **kwargs)

        self.hyperparameters = self.addGroupBox(title='Hyperparameters',
                                                parameters=[
                                                    tool.parameters.epochs,
                                                    tool.parameters.batch_size,
                                                    tool.parameters.optimizer,
                                                    tool.parameters.learning_rate,
                                                    tool.parameters.focal_gamma,
                                                    tool.parameters.seed,
                                                    ],
                                                widget_args={
                                                    'learning_rate': {'decimals': 5, 'single_step': 1e-5, 'adaptive': False, },
                                                    }
                                                )

        self.scheduler = self.addGroupBox(title='Scheduler',
                                          parameters=[
                                              tool.parameters.step_scheduler,
                                              tool.parameters.step_gamma,
                                              tool.parameters.step_size,
                                              tool.parameters.warmup,
                                              tool.parameters.wu_start,
                                              tool.parameters.wu_end,
                                              tool.parameters.wu_duration,
                                              tool.parameters.reduce_lr_on_plateau,
                                              tool.parameters.reduce_patience,
                                              tool.parameters.reduction_factor,
                                              ],
                                          )

        self.regularization = self.addGroupBox(title='Regularization',
                                               parameters=[
                                                   tool.parameters.regularization,
                                                   tool.parameters.lambda_reg,
                                                   ],
                                               widget_args={
                                                   'lambda_reg': {'decimals': 5, 'single_step': 1e-5, 'adaptive': False},
                                                   }
                                               )

        self.datasets = self.addGroupBox(title='Datasets',
                                         parameters=[
                                             tool.parameters.num_training,
                                             tool.parameters.num_validation,
                                             tool.parameters.num_test,
                                             tool.parameters.data_seed,
                                             tool.parameters.augmentation,
                                             # tool.parameters.idx,
                                             ],
                                         widget_args={
                                             'num_test': {'enabled': False},
                                             }
                                         )

        self.tool.parameters.hdf5_file.current_dir = PyDetecDiv.tools['cnrs.plewniak.roiseqhdf5creator'].working_dir
        self.hdf5_file = self.addGroupBox(title='Data File',
                                          parameters=[
                                              tool.parameters.hdf5_file,
                                              tool.parameters.time_first,
                                              ],
                                          )

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.hyperparameters,
                             self.regularization,
                             self.scheduler,
                             self.datasets, self.hdf5_file,
                             self.button_box])

        set_connections({self.button_box.accepted              : lambda: self.wait_for_process(tool.model_trainer.train_model),
                         self.button_box.rejected              : lambda: print('Rejected'),
                         tool.parameters.epochs.changed        : lambda: print(tool.parameters.epochs.value),
                         tool.parameters.optimizer.changed     : lambda: print(tool.parameters.optimizer.value),
                         tool.parameters.num_training.changed  : lambda: self.update_datasets(tool.parameters.num_training),
                         tool.parameters.num_validation.changed: lambda: self.update_datasets(tool.parameters.num_validation),
                         })

        self.run_after_process([plot_training_results, tool.dump_train_stats])

        tool.parameters.reset()
        self.fit_to_contents()
        self.exec()

    def update_datasets(self, changed_param: FloatParameter = None) -> None:
        """
        Update the dataset proportions values to make sure they sup up to 1

        :param changed_param: the changed parameter
        """
        self.tool.parameters.num_test.value = 1.0 - (self.tool.parameters.num_training + self.tool.parameters.num_validation)
        if changed_param:
            total = self.tool.parameters.num_test + self.tool.parameters.num_training + self.tool.parameters.num_validation
            if total > 1.0:
                changed_param.value = changed_param - total + 1.0


class TrainModelAction(ToolAction):
    """
    Action to import raw data images into a project
    """

    def __init__(self, tool_name: str, parent: QMenu = None):
        super().__init__("Train model", tool_name, parent)

    def determine_enabled_status(self, **kwargs: dict[str, Any]):
        self.setEnabled(False)
        if project_exists(PyDetecDiv.project_name):
            with pydetecdiv_project(PyDetecDiv.project_name) as project:
                if (project.count_objects('RoiAnnotations') > 0) and (project.count_objects('Classification') > 0):
                    self.setEnabled(True)

    def launch(self):
        """
        Run training procedure
        """
        TrainModelDialog(self.tool)
