from typing import Any

from pydetecdiv.app.gui.tools import ToolMenu, ToolAction
from pydetecdiv.app.gui.tools.data.format.convert_to_ndtiff import Convert2NDTiffDialog


class DataFormatMenu(ToolMenu):
    def __init__(self, tool_name: str, enable = None, **kwargs: dict[str, Any]):
        super().__init__(tool_name, enable=enable, **kwargs)
        ToolAction(tool_name, 'convert2ndtiff', Convert2NDTiffDialog, self)
