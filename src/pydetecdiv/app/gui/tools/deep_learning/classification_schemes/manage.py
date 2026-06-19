import json
from typing import Any

import polars
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog
from pyqtgraph.debug import pretty

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv, MessageDialog
from pydetecdiv.app.gui.core.widgets import set_connections, TableView
from pydetecdiv.app.models import EditableTableModel
from pydetecdiv.app.gui.tools import ToolDialog
from pydetecdiv.domain.Classification import Classification
from pydetecdiv.domain.tools.deep_learning.classification_schemes import ClassificationSchemeManagement


class ManageClassificationSchemeDialog(ToolDialog):
    def __init__(self, tool: ClassificationSchemeManagement, title: str | None = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(tool, title, **kwargs)
        self.setMinimumWidth(650)

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.data_view = TableView(self,EditableTableModel(project.get_polars('Classification')))

        button_box = self.addButtonBox()
        self.edit_button = button_box.addButton('Edit', QDialogButtonBox.ButtonRole.ActionRole)
        new_button = button_box.addButton('New', QDialogButtonBox.ButtonRole.ActionRole)
        self.delete_button = button_box.addButton('Delete', QDialogButtonBox.ButtonRole.ActionRole)
        self.export_button = button_box.addButton('Export', QDialogButtonBox.ButtonRole.ActionRole)
        import_button = button_box.addButton('Import', QDialogButtonBox.ButtonRole.ActionRole)

        self.toggle_buttons()

        self.arrangeWidgets([
            self.data_view,
            button_box
            ])

        set_connections({
            button_box.accepted: self.delete_removed_schemes,
            self.edit_button.pressed: self.edit_selected_schemes,
            new_button.pressed: self.add_new_scheme,
            self.delete_button.pressed: self.delete_selected_schemes,
            self.export_button.pressed: self.export,
            import_button.pressed: self.import_json,
            })

        self.fit_to_contents()
        self.exec()

    def toggle_buttons(self):
        if self.data_view.is_empty():
            self.edit_button.setEnabled(False)
            self.delete_button.setEnabled(False)
        else:
            self.edit_button.setEnabled(True)
            self.delete_button.setEnabled(True)

    def edit_selected_schemes(self):
        for row in self.data_view.selected_rows(data=True).iter_rows(named=True):
            if ClassificationSchemeManagement.scheme_is_not_used(row['name']):
                EditClassificationSchemeDialog(self.tool, self, title=f'Edit {row["name"]}', row=row)
            else:
                MessageDialog(f'{row["name"]} classification scheme cannot be edited because it is already in use.',)

    def delete_selected_schemes(self):
        for row in self.data_view.selected_rows(data=True).iter_rows(named=True):
            if ClassificationSchemeManagement.scheme_is_not_used(row['name']):
                self.data_view.delete_row(row['id_'])
                self.toggle_buttons()
            else:
                MessageDialog(f'{row["name"]} classification scheme cannot be deleted because it is in use.',)

    def add_new_scheme(self):
        EditClassificationSchemeDialog(self.tool, self, title='Add new scheme', row=None)
        self.toggle_buttons()

    def delete_removed_schemes(self):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            for row in project.get_polars('Classification').join(self.data_view.data,
                                                                 left_on='name', right_on='name', how='anti').iter_rows(named=True):
                project.delete(project.get_object('Classification', row['id_']))
        self.close()

    def refresh(self):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.data_view.set_data(project.get_polars('Classification'))
        self.toggle_buttons()

    def save_schemes(self):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            for row in self.data_view.data.iter_rows(named=True):
                scheme = project.get_named_object('Classification', row['name'])
                if scheme is None:
                    scheme = Classification(project=project, name=row['name'], classes=row['classes'], key_val={})
                    project.save(scheme)
                else:
                    if ClassificationSchemeManagement.scheme_is_not_used(row['name']):
                        scheme.classes = row['classes']
                        scheme.validate(updated=True)
                    else:
                        MessageDialog(f'{row["name"]} classification scheme cannot be updated because it is in use.',)

    def export(self):
        filters = ["All files (*)", "JSON (*.json *.jsn)", ]
        file_name, _ = QFileDialog.getSaveFileName(self, caption='Choose file', dir=self.tool.working_dir, filter=";;".join(filters),
                                                       selectedFilter="JSON (*.json *.jsn)")
        if file_name:
            schemes = self.data_view.selected_rows(data=True)
            schemes.write_json(file_name, )

    def import_json(self):
        filters = ["All files (*)", "JSON (*.json *.jsn)", ]
        file_name, _ = QFileDialog.getOpenFileName(self, caption='Choose file', dir=self.tool.working_dir, filter=";;".join(filters),
                                                       selectedFilter="JSON (*.json *.jsn)")
        if file_name:
            imported_schemes = polars.read_json(file_name, infer_schema_length=5)
            if self.data_view.is_empty():
                self.data_view.set_data(imported_schemes)
            else:
                print(imported_schemes.join(self.data_view.data, left_on='name', right_on='name', how='semi'))
                print(imported_schemes.join(self.data_view.data, left_on='name', right_on='name', how='anti'))
                self.data_view.add_rows(imported_schemes.join(self.data_view.data, left_on='name', right_on='name', how='semi'))
                self.data_view.add_rows(imported_schemes.join(self.data_view.data, left_on='name', right_on='name', how='anti'))
        self.save_schemes()
        self.refresh()


class EditClassificationSchemeDialog(ToolDialog):
    def __init__(self, tool: ClassificationSchemeManagement, parent: ManageClassificationSchemeDialog, title: str | None = None,
                 row: dict[str, Any]|None = None, **kwargs: dict[str, Any]) -> None:
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
