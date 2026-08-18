from typing import Any

from pydetecdiv.utils import check
from pydetecdiv.app.gui.tools import ToolMenu, ToolAction
from pydetecdiv.app.gui.tools.video_classifier.train import TrainModelDialog


class VideoClassifierMenu(ToolMenu):
    def __init__(self, tool_name: str, **kwargs: dict[str, Any]):
        super().__init__(tool_name, **kwargs)
        ToolAction(tool_name, 'train_model', TrainModelDialog, self,
                   enable=check.AND(check.if_annotations, check.if_exists_roi_hdf5))
