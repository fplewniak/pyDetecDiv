from typing import TYPE_CHECKING

from pydetecdiv.app.gui.core.widgets import ParametersFormGroupBox, set_connections
from pydetecdiv.app.gui.tools import ToolAction, ToolDialog
from pydetecdiv.app.tools.deep_learning import DeepTool

if TYPE_CHECKING:
    from pydetecdiv.app.gui.tools.video_classifier import VideoClassifierMenu


class TrainModelDialog(ToolDialog):
    def __init__(self, tool: DeepTool, **kwargs):
        super().__init__(tool, title='Training Video classifier', **kwargs)

        self.hyperparameters = self.addGroupBox(title='Hyperparameters',
                                                parameters=[
                                                    tool.parameters['epochs'],
                                                    tool.parameters['batch_size'],
                                                    tool.parameters['optimizer'],
                                                    tool.parameters['seed'],
                                                    ])

        self.datasets = self.addGroupBox(title='Datasets',
                                         parameters=[
                                             tool.parameters['num_training'],
                                             tool.parameters['num_validation'],
                                             tool.parameters['num_test'],
                                             tool.parameters['data_seed'],
                                             ])

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.hyperparameters, self.datasets, self.button_box])

        set_connections({self.button_box.accepted: tool.model_trainer.train_model,
                         self.button_box.rejected: lambda: print('Rejected'),
                         tool.parameters['epochs'].changed: lambda: print(tool.parameters['epochs'].value),
                         tool.parameters['optimizer'].changed: lambda: print(tool.parameters['optimizer'].value),
                         tool.parameters['num_training'].changed: lambda: print(tool.parameters['num_training'].value),
                         })

        self.fit_to_contents()
        self.exec()


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
        # print(self.parent().title())
        # print(self.parent().tool.name, self.parent().tool.version, self.parent().tool.id_, )
        TrainModelDialog(self.parent().tool)
