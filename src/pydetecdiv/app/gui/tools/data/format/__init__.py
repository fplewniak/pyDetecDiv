"""
Data format tool classes for GUI
"""
from typing import Any, Callable

from pydetecdiv.app.gui.tools import ToolMenu, ToolAction
from pydetecdiv.app.gui.tools.data.format.convert_to_ndtiff import Metadata2NDTiffDialog


class DataFormatMenu(ToolMenu):
    """
    Menu for data format tool
    """
    def __init__(self, tool_name: str, enable: Callable[..., Any] | None = None, **kwargs: dict[str, Any]):
        super().__init__(tool_name, enable=enable, **kwargs)
        ToolAction(tool_name, 'metadata2ndtiff', Metadata2NDTiffDialog, self)
