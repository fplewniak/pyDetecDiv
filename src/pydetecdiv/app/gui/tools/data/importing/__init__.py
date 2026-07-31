"""
Data import GUI
"""
from typing import Any

from pydetecdiv.app.gui import Enable
from pydetecdiv.app.gui.tools import ToolMenu, ToolAction
from pydetecdiv.app.gui.tools.data.importing.import_dialog import DataImportDialog, AnnotatedROIsImportDialog


class DataImportMenu(ToolMenu):
    def __init__(self, tool_name: str, enable = None, **kwargs: dict[str, Any]):
        super().__init__(tool_name, enable=enable, **kwargs)
        ToolAction(tool_name, 'import_images', DataImportDialog, self)
        ToolAction(tool_name, 'import_annotated_rois', AnnotatedROIsImportDialog, self, enable = Enable.if_class_scheme)
