from typing import Any

from PySide6.QtWidgets import QAbstractItemView

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.app.gui.core.widgets import set_connections, TableView
from pydetecdiv.app.models import TableModel
from pydetecdiv.app.tools import Tool
from pydetecdiv.app.gui.tools import ToolDialog


class ManageClassificationSchemeDialog(ToolDialog):
    def __init__(self, tool: Tool, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(tool, title, **kwargs)

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.data_view = TableView(self,TableModel(project.get_polars('Classification')))

        classification_management = self.addGroupBox(
                parameters=[
                    self.tool.parameters.name,
                    ],
                )

        button_box = self.addButtonBox()

        self.arrangeWidgets([
            self.data_view,
            classification_management,
            button_box
            ])

        set_connections({
            button_box.accepted: self.edit_selected_scheme,
            })

        self.fit_to_contents()
        self.exec()

    def edit_selected_scheme(self):
        print(self.data_view.selected_rows(data=True))
