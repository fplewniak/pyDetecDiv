"""
GUI for deep learning model summary tool
"""
import json
from typing import Any

from torchinfo import summary

from pydetecdiv.app import set_connections
from pydetecdiv.app.tools import Tool
from pydetecdiv.app.gui.tools import ToolDialog


class ModelSummaryDialog(ToolDialog):
    """
    Dialog window for deep learning model summary tool
    """
    def __init__(self, tool: Tool, title: str | None = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(tool, title, **kwargs)

        model_choice = self.addGroupBox(
                parameters=[
                    self.tool.parameters.model,
                    self.tool.parameters.num_classes,
                    self.tool.parameters.layers,
                    self.tool.parameters.strides,
                    self.tool.parameters.batch_size,
                    self.tool.parameters.depth,
                    ],
                )
        button_box = self.addButtonBox()

        self.arrangeWidgets([
            model_choice,
            button_box
            ])

        set_connections({
            button_box.accepted: self.tool.callback,
            })

        self.fit_to_contents()
        self.exec()
