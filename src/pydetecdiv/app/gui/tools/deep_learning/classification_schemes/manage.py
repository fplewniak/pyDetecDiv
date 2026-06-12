from typing import Any

from PySide6.QtWidgets import QDialogButtonBox

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv, MessageDialog
from pydetecdiv.app.gui.core.widgets import set_connections, TableView
from pydetecdiv.app.models import TableModel
from pydetecdiv.app.tools import Tool
from pydetecdiv.app.gui.tools import ToolDialog


class ManageClassificationSchemeDialog(ToolDialog):
    def __init__(self, tool: Tool, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(tool, title, **kwargs)
        self.setMinimumWidth(650)

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.data_view = TableView(self,TableModel(project.get_polars('Classification')))

        button_box = self.addButtonBox()
        new_button = button_box.addButton('New', QDialogButtonBox.ButtonRole.ActionRole)

        self.arrangeWidgets([
            self.data_view,
            button_box
            ])

        set_connections({
            button_box.accepted: self.edit_selected_schemes,
            new_button.pressed: self.add_new_scheme,
            })

        self.fit_to_contents()
        self.exec()

    def edit_selected_schemes(self):
        for row in self.data_view.selected_rows(data=True).iter_rows(named=True):
            if self.tool.scheme_is_not_used(row['name']) :
                EditClassificationSchemeDialog(self.tool, self, title=f'Edit {row["name"]}', row=row)
            else:
                MessageDialog(f'{row["name"]} classification scheme cannot be edited because it is already in use.',)

    def add_new_scheme(self):
        EditClassificationSchemeDialog(self.tool, self, title='Add new scheme', row=None)

    def refresh(self):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.data_view._model.set_data(project.get_polars('Classification'))


class EditClassificationSchemeDialog(ToolDialog):
    def __init__(self, tool: Tool, parent: ManageClassificationSchemeDialog, title: str = None, row: dict[str, Any]|None = None,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(tool, title, **kwargs)
        self.parent = parent

        if row is not None:
            self.tool.parameters.name.set_value(row['name'])
            self.tool.parameters.classes.set_value(row['classes'])

        classification_management = self.addGroupBox(
                parameters=[
                    self.tool.parameters.name,
                    self.tool.parameters.classes,
                    ],
                )

        button_box = self.addButtonBox()
        new_class_button = button_box.addButton('Add class', QDialogButtonBox.ButtonRole.ActionRole)

        self.arrangeWidgets([
            classification_management,
            button_box
            ])

        set_connections({
            button_box.accepted: self.save_edit,
            new_class_button.pressed: self.add_new_class,
            })

        self.fit_to_contents()
        self.exec()

    def add_new_class(self):
        self.tool.parameters.classes.append('new')


    def save_edit(self):
        self.tool.save_scheme()
        self.parent.refresh()
        self.close()
