from typing import TYPE_CHECKING

from pydetecdiv.app.gui.core.widgets import ParametersFormGroupBox
from pydetecdiv.app.gui.tools import ToolAction, ToolDialog

if TYPE_CHECKING:
    from pydetecdiv.app.gui.tools.video_classifier import VideoClassifierMenu


class TrainModelDialog(ToolDialog):
    def __init__(self, tool, **kwargs):
        super().__init__(tool, title='Training Video classifier', **kwargs)

        self.classifier = self.addGroupBox('Classifier')
        self.classifier.addOption(tool.parameters['seed'])
        self.expandable = self.classifier.addSubBox(ParametersFormGroupBox, expandable=True, show=False, title='Seed')
        self.expandable.addOption(tool.parameters['seed'])

        self.arrangeWidgets([self.classifier])
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
        print(self.parent().title())
        print(self.parent().tool.name, self.parent().tool.version, self.parent().tool.id_, )
        TrainModelDialog(self.parent().tool)
