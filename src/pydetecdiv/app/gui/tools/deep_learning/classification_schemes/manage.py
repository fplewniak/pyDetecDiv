from typing import Any

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

        button_box = self.addButtonBox()

        self.arrangeWidgets([
            self.data_view,
            button_box
            ])

        set_connections({
            button_box.accepted: self.edit_selected_schemes,
            })

        self.fit_to_contents()
        self.exec()

    def edit_selected_schemes(self):
        for row in self.data_view.selected_rows(data=True).iter_rows(named=True):
            EditClassificationSchemeDialog(self.tool, self, title=f'Edit {row["name"]}', row=row)

    def refresh(self):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.data_view._model.set_data(project.get_polars('Classification'))


class EditClassificationSchemeDialog(ToolDialog):
    def __init__(self, tool: Tool, parent: ManageClassificationSchemeDialog, title: str = None, row: dict[str, Any]|None = None,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(tool, title, **kwargs)
        self.parent = parent

        self.tool.parameters.name.set_value(row['name'])
        self.tool.parameters.classes.set_value(row['classes'])

        classification_management = self.addGroupBox(
                parameters=[
                    self.tool.parameters.name,
                    self.tool.parameters.classes,
                    ],
                )

        button_box = self.addButtonBox()

        self.arrangeWidgets([
            classification_management,
            button_box
            ])

        set_connections({
            button_box.accepted: self.save_edit,
            })

        self.fit_to_contents()
        self.exec()

    def save_edit(self):
        self.tool.save_scheme()
        self.parent.refresh()
        self.close()
