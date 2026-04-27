from typing import Any

from pydetecdiv.app.gui.tools import ToolMenu
from pydetecdiv.app.gui.tools.deep_learning.model_info.summary import ModelSummaryAction
from pydetecdiv.domain.tools.deep_learning.model_info import ModelInfo


class ModelInfoMenu(ToolMenu):
    def __init__(self, tool: ModelInfo, **kwargs: dict[str, Any]):
        super().__init__(tool, **kwargs)
        ModelSummaryAction(self)
