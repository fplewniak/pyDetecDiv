from typing import Any

from pydetecdiv.app.gui.tools import ToolMenu, ToolAction
from pydetecdiv.app.gui.tools.deep_learning.model_info.summary import ModelSummaryDialog


class ModelInfoMenu(ToolMenu):
    def __init__(self, tool_name: str, **kwargs: dict[str, Any]):
        super().__init__(tool_name, **kwargs)
        ToolAction('Show model summary', tool_name, ModelSummaryDialog, self)
