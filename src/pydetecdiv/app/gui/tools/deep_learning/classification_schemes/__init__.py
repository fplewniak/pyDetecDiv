from typing import Any

from pydetecdiv.app.gui import Enable
from pydetecdiv.app.gui.tools import ToolMenu, ToolAction
from pydetecdiv.app.gui.tools.deep_learning.classification_schemes.manage import ManageClassificationSchemeDialog


class ClassificationSchemeMenu(ToolMenu):
    def __init__(self, tool_name: str, **kwargs: dict[str, Any]):
        super().__init__(tool_name, **kwargs)
        ToolAction(tool_name, 'manage_schemes', ManageClassificationSchemeDialog, self, enable=Enable.if_project_exists)
