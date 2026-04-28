from typing import Any

from pydetecdiv.app.gui.tools import ToolMenu
from pydetecdiv.app.gui.tools.deep_learning.classification_schemes.manage import ManageClassificationSchemesAction
from pydetecdiv.domain.tools.deep_learning.classification_schemes import ClassificationSchemeManagement


class ClassificationSchemeMenu(ToolMenu):
    def __init__(self, tool: ClassificationSchemeManagement, **kwargs: dict[str, Any]):
        super().__init__(tool, **kwargs)
        ManageClassificationSchemesAction(tool, self)
