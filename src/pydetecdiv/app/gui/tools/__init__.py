"""
Generic widgets providing basic functionalities to build tool GUIs. These widgets are expected to be extended and implemented
to meet the specific needs of tools
"""

from typing import Any

from pydetecdiv.app.gui.core.widgets import Dialog
from pydetecdiv.app.tools import Tool


class ToolDialog(Dialog):
    """
    Generic tool dialog window
    """
    def __init__(self, tool: Tool, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(title, **kwargs)
        self.tool = tool
