"""
Data edition tool classes for GUI
"""
from typing import Any, Callable

from pydetecdiv.app.gui import Enable
from pydetecdiv.app.gui.tools import ToolMenu, ToolAction
from pydetecdiv.app.gui.tools.data.edition.drift_correction import ComputeDriftDialog


class DriftCorrectionMenu(ToolMenu):
    """
    Menu for data format tool
    """
    def __init__(self, tool_name: str, enable: Callable[..., Any] | None = None, **kwargs: dict[str, Any]):
        super().__init__(tool_name, enable=enable, **kwargs)
        ToolAction(tool_name, 'compute_drift', ComputeDriftDialog, self, enable=Enable.if_image_resources)
