from typing import Any, TYPE_CHECKING

from pydetecdiv.app.gui.core.widgets import set_connections
from pydetecdiv.app.tools import Tool
from pydetecdiv.app.gui.tools import ToolDialog, ToolAction

if TYPE_CHECKING:
    from pydetecdiv.app.gui.tools.deep_learning.classification_schemes import ClassificationSchemeMenu


class ManageClassificationSchemeDialog(ToolDialog):
    def __init__(self, tool: Tool, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(tool, title, **kwargs)

        classification_management = self.addGroupBox(
                parameters=[
                    self.tool.parameters.name,
                    ],
                )

        button_box = self.addButtonBox()

        self.arrangeWidgets([
            classification_management,
            button_box
            ])

        set_connections({
            button_box.accepted: lambda: print(self.tool.parameters),
            })

        self.fit_to_contents()
        self.exec()


class ManageClassificationSchemesAction(ToolAction):
    """
    Action to open a shared data source configuration window
    """

    def __init__(self, parent: 'ClassificationSchemeMenu'):
        super().__init__("Manage classification schemes", parent)
        # TODO check a project is open
        self.setEnabled(True)

    def launch(self):
        """
        Run training procedure
        """
        ManageClassificationSchemeDialog(self.parent().tool)
