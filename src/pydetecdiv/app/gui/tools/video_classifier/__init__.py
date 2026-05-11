from typing import Any

from pydetecdiv.app.gui import ToolAction, Enable
from pydetecdiv.app.gui.tools import ToolMenu
from pydetecdiv.app.gui.tools.video_classifier.train import TrainModelDialog


class VideoClassifierMenu(ToolMenu):
    def __init__(self, tool_name: str, **kwargs: dict[str, Any]):
        super().__init__(tool_name, **kwargs)
        ToolAction('Train model', tool_name, TrainModelDialog, self, enable=Enable.if_annotations)
