from typing import TYPE_CHECKING

from pydetecdiv.app.gui.core.widgets import ParametersFormGroupBox, set_connections
from pydetecdiv.app.gui.tools import ToolAction, ToolDialog
from pydetecdiv.app.tools.deep_learning import DeepTool

if TYPE_CHECKING:
    from pydetecdiv.app.gui.tools.video_classifier import VideoClassifierMenu


class TrainModelDialog(ToolDialog):
    def __init__(self, tool: DeepTool, **kwargs):
        super().__init__(tool, title='Training Video classifier', **kwargs)

        self.classifier = self.addGroupBox(title='Classifier',
                                           parameters=[
                                               tool.parameters['seed'],
                                               tool.parameters['check'],
                                               tool.parameters['text'],
                                               tool.parameters['choice']
                                               ])

        self.expandable = self.classifier.addSubBox(ParametersFormGroupBox, expandable=True, show=False, title='More options',
                                                    parameters=[
                                                        tool.parameters['choice'],
                                                        tool.parameters['text'],
                                                        tool.parameters['seed'],
                                                        tool.parameters['check'],
                                                        ])

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.classifier, self.button_box])

        set_connections({self.button_box.accepted    : tool.model_trainer.train_model,
                         self.button_box.rejected    : lambda: print('Rejected')
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
