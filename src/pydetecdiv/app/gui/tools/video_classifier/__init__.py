from typing import Any

from pydetecdiv.app.gui.tools import ToolMenu
from pydetecdiv.app.gui.tools.video_classifier.train import TrainModelAction
from pydetecdiv.domain.tools.video_classifier import VideoClassifier


class VideoClassifierMenu(ToolMenu):
    def __init__(self, tool: VideoClassifier, **kwargs: dict[str, Any]):
        super().__init__(tool, **kwargs)
        TrainModelAction(self)
