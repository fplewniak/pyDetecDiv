from typing import TYPE_CHECKING

from pydetecdiv.app.gui.tools import ToolAction
if TYPE_CHECKING:
    from pydetecdiv.app.gui.tools.video_classifier import VideoClassifierMenu


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
        print(self.parent().tool.name, self.parent().tool.version, self.parent().tool.id_,)
