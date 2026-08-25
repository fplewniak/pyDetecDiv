"""
Data edition tool classes for GUI
"""
from typing import Any, Callable

from pydetecdiv.utils import check
from pydetecdiv.app.gui.tools import ToolMenu, ToolAction
from pydetecdiv.app.gui.tools.data.edition.drift_correction import ComputeDriftDialog


class DriftCorrectionMenu(ToolMenu):
    """
    Menu for data format tool
    """
    def __init__(self, tool_name: str, enable: Callable[..., Any] | None = None, **kwargs: dict[str, Any]):
        super().__init__(tool_name, enable=enable, **kwargs)
        ToolAction(tool_name, 'compute_drift', ComputeDriftDialog, self, enable=check.if_image_resources)
        ToolAction(tool_name, 'apply_drift_correction', self.tool.toggle_drift_correction, self,
                   enable=check.if_drift_correction, checkable=True)
