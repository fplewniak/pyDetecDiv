"""
Data import GUI
"""
from typing import Any

from pydetecdiv.app.gui.Enable import if_project_exists, if_class_scheme, if_image_resources, AND
from pydetecdiv.app.gui.tools import ToolMenu, ToolAction
from pydetecdiv.app.gui.tools.data.importing.import_dialog import DataImportDialog, AnnotatedROIsImportDialog


class DataImportMenu(ToolMenu):
    def __init__(self, tool_name: str, enable = None, **kwargs: dict[str, Any]):
        super().__init__(tool_name, enable=enable, **kwargs)
        ToolAction(tool_name, 'import_images', DataImportDialog, self, enable = if_project_exists)
        ToolAction(tool_name, 'import_annotated_rois', AnnotatedROIsImportDialog, self,
                   enable = AND(if_class_scheme, if_image_resources))
