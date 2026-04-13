from typing import TYPE_CHECKING

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets import ParametersFormGroupBox, set_connections
from pydetecdiv.app.gui.tools import ToolAction, ToolDialog
from pydetecdiv.app.parameters import FloatParameter
from pydetecdiv.app.tools.deep_learning import DeepTool

if TYPE_CHECKING:
    from pydetecdiv.app.gui.tools.video_classifier import VideoClassifierMenu


class TrainModelDialog(ToolDialog):
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
                                                    'learning_rate': {'decimals': 5, 'single_step': 1e-5, 'adaptive': False,},
                                                    }
                                                )

        self.datasets = self.addGroupBox(title='Datasets',
                                         parameters=[
                                             tool.parameters.num_training,
                                             tool.parameters.num_validation,
                                             tool.parameters.num_test,
                                             tool.parameters.data_seed,
                                             ],
                                         widget_args={
                                             'num_test': {'decimals': 2},
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

        self.arrangeWidgets([self.hyperparameters, self.datasets, self.hdf5_file, self.button_box])

        set_connections({self.button_box.accepted: tool.model_trainer.train_model,
                         self.button_box.rejected: lambda: print('Rejected'),
                         tool.parameters.epochs.changed: lambda: print(tool.parameters.epochs.value),
                         tool.parameters.optimizer.changed: lambda: print(tool.parameters.optimizer.value),
                         tool.parameters.num_training.changed: lambda: self.update_datasets(tool.parameters.num_training),
                         tool.parameters.num_validation.changed: lambda: self.update_datasets(tool.parameters.num_validation),
                         })

        self.fit_to_contents()
        self.exec()

    def update_datasets(self, changed_param: FloatParameter = None):
        self.tool.parameters.num_test.value = 1.0 - (self.tool.parameters.num_training + self.tool.parameters.num_validation)
        if changed_param:
            total = self.tool.parameters.num_test + self.tool.parameters.num_training + self.tool.parameters.num_validation
            if total > 1.0:
                changed_param.value = changed_param - total + 1.0


class TrainModelAction(ToolAction):
    """
    Action to import raw data images into a project
    """

    def __init__(self, parent: 'VideoClassifierMenu'):
        super().__init__("Train model", parent)
        self.setEnabled(True)

    def launch(self):
        """
        Run training procedure
        """
        # TODO check there are annotated ROIs in the database. This will be conveniently done using a new Annotations table with
        # TODO the count_objects() method
        TrainModelDialog(self.parent().tool)
