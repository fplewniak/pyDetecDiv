from typing import Any, TYPE_CHECKING

from torchinfo import summary

from pydetecdiv.app.tools import Tool
from pydetecdiv.app.gui.tools import ToolDialog, ToolAction
from pydetecdiv.plugins.gui import set_connections

if TYPE_CHECKING:
    from pydetecdiv.app.gui.tools.deep_learning.model_info import ModelInfoMenu


class ModelSummaryDialog(ToolDialog):
    def __init__(self, tool: Tool, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(tool, title, **kwargs)

        model_choice = self.addGroupBox(
                parameters=[
                    self.tool.parameters.model,
                    self.tool.parameters.num_classes,
                    self.tool.parameters.batch_size
                    ],
                )
        button_box = self.addButtonBox()

        self.arrangeWidgets([
            model_choice,
            button_box
            ])

        set_connections({
            button_box.accepted: self.show_model_information,
            })

        self.fit_to_contents()
        self.exec()

    def show_model_information(self):
        model = self.tool.parameters.model.value(n_classes=self.tool.parameters.num_classes.value)
        summary(model, (self.tool.parameters.batch_size.value,) + model.expected_shape[1:], device='cpu')


class ModelSummaryAction(ToolAction):
    """
    Action to open a shared data source configuration window
    """

    def __init__(self, parent: 'ModelInfoMenu'):
        super().__init__("Show model summary", parent)
        # TODO check a project is open
        self.setEnabled(True)

    def launch(self):
        """
        Run training procedure
        """
        ModelSummaryDialog(self.parent().tool)
