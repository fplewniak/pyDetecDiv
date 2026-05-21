import json
from typing import Any

from torchinfo import summary

from pydetecdiv.app.tools import Tool
from pydetecdiv.app.gui.tools import ToolDialog
from pydetecdiv.plugins.gui import set_connections


class ModelSummaryDialog(ToolDialog):
    def __init__(self, tool: Tool, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(tool, title, **kwargs)

        model_choice = self.addGroupBox(
                parameters=[
                    self.tool.parameters.model,
                    self.tool.parameters.num_classes,
                    self.tool.parameters.layers,
                    self.tool.parameters.strides,
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
        if self.tool.parameters.model.key == 'CustomR2Plus_1D':
            model = self.tool.parameters.model.value(n_classes=self.tool.parameters.num_classes.value,
                                                     layers=json.loads(self.tool.parameters.layers.value),
                                                     strides=json.loads(self.tool.parameters.strides.value),)
        else:
            model = self.tool.parameters.model.value(n_classes=self.tool.parameters.num_classes.value)
        summary(model, (self.tool.parameters.batch_size.value,) + model.expected_shape[1:], device='cpu')
