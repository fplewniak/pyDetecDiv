import os
from typing import Self

import polars
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QHeaderView, QVBoxLayout, QSizePolicy, QDialogButtonBox, QTableView, QDialog, QPushButton, QLineEdit,
                               QGroupBox, QHBoxLayout, QFileDialog, QWidget, QLabel)

import pydetecdiv.settings
from pydetecdiv.app.models import TableModel, EditableTableModel
from pydetecdiv.settings import Device, datapath_list, datapath_file


class TableEditor(QDialog):
    """
    A dialog window displaying a table of source paths and allowing configuration of those that are not yet defined
    """

    def __init__(self, title=None, description='<b>Source paths defined on other devices:</b>', force_resolution=False,
                 editable_col=None,
                 **kwargs):
        super().__init__(**kwargs)
        if title is not None:
            self.setWindowTitle(title)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.path_id = None
        self.path_name = None
        self.editable_col = editable_col

        table_label = QLabel(self)
        table_label.setText(description)

        self.model = EditableTableModel(polars.DataFrame([{'name': '', 'device': '', 'path': '', 'path_id': ''}]),
                                        editable_col=self.editable_col)
        self.table_view = QTableView(self)
        self.table_view.setModel(self.model)

        # # QTableView Headers
        self.horizontal_header = self.table_view.horizontalHeader()
        self.vertical_header = self.table_view.verticalHeader()
        self.horizontal_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.vertical_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontal_header.setStretchLastSection(True)

        path_group_label = QLabel(self)
        path_group_label.setText('<b>Define corresponding local path:</b>')
        path_group = QGroupBox(self)
        # path_group.setTitle('<b>Define corresponding local path:</b>')
        name_label = QLabel('name')
        self.name_edit = QLineEdit(path_group)
        path_label = QLabel('path')
        self.path_edit = QLineEdit(path_group)
        button_path = QPushButton(path_group)
        icon = QIcon(":icons/file_chooser")
        button_path.setIcon(icon)
        button_path.clicked.connect(self.select_path)
        self.path_edit.textChanged.connect(self.path_edit_changed)
        self.name_edit.textChanged.connect(self.path_edit_changed)

        if force_resolution:
            self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        else:
            self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Close)
            self.button_box.rejected.connect(self.close)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.button_box.accepted.connect(self.save_local_datapath)

        self.main_layout = QVBoxLayout(self)
        # table_layout = QHBoxLayout(table_group)
        path_layout = QHBoxLayout(path_group)

        # table_layout.addWidget(self.table_view)
        path_layout.addWidget(name_label)
        path_layout.addWidget(self.name_edit)
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(button_path)

        # self.main_layout.addWidget(table_group)
        self.main_layout.addWidget(table_label)
        self.main_layout.addWidget(self.table_view)
        self.main_layout.addWidget(path_group_label)
        self.main_layout.addWidget(path_group)
        self.main_layout.addWidget(self.button_box)

    def set_data(self, data: polars.DataFrame) -> Self:
        """
        Sets the data model

        :param data: the data to display in the table
        """
        self.path_id = data.select('path_id').unique().item()
        self.name_edit.setText(data.select('name').unique().item())
        # self.model = TableModel(data.select(['name', 'device', 'path']))
        self.model = EditableTableModel(data, editable_col=self.editable_col)
        self.table_view.setModel(self.model)
        return self

    def hide_columns(self, columns: int | list[int]) -> None:
        """
        Hides the specified column(s)

        :param columns: the index or list of indices for the columns to hide
        """
        if not isinstance(columns, list):
            columns = [columns]
        for column in columns:
            self.table_view.setColumnHidden(self.model.df.columns.index(column), True)

    def show_columns(self, columns: int | list[int]) -> None:
        """
        Shows the specified column(s)

        :param columns: the index or list of indices for the columns to show
        """
        if not isinstance(columns, list):
            columns = [columns]
        for column in columns:
            self.table_view.setColumnHidden(self.model.df.columns.index(column), False)

    def select_path(self) -> None:
        """
        Opens a Filedialog window to select a path
        """
        dir_name = '/'
        if dir_name != self.path_edit.text() and self.path_edit.text():
            dir_name = self.path_edit.text()
        directory = QFileDialog.getExistingDirectory(self, caption='Choose data source directory', dir=dir_name,
                                                     options=QFileDialog.Option.ShowDirsOnly)
        if directory:
            self.path_edit.setText(directory)

    def path_edit_changed(self) -> None:
        """
        Each time the path or name line edit have been changed, the Ok button is enabled if both are specified or disabled
        otherwise. Note that the path must contain a valid path to enable the Ok button
        """
        if os.path.isdir(self.path_edit.text()) and self.name_edit.text():
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def save_local_datapath(self) -> None:
        """
        Save the data source path to the configuration file in the workspace directory
        """
        data = datapath_list()
        local_path = polars.DataFrame({
            'name'   : self.name_edit.text(),
            'path_id': self.path_id,
            'device' : Device.name(),
            'MAC'    : Device.mac(),
            'path'   : self.path_edit.text()
            })
        data.extend(local_path).sort(by=['name', 'device']).write_csv(datapath_file())
        self.close()


class PathCreator(TableEditor):
    """
    A TableEditor for new source path definition. The name is not displayed in the table as it has not been defined yet.
    """

    def __init__(self, title=None, description='Define new path on this device:', **kwargs):
        super().__init__(title, description, **kwargs)
        if title is not None:
            self.setWindowTitle(title)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.path_id = None
        self.path_name = None
        self.path = None

    def set_data(self, data: polars.DataFrame) -> Self:
        """
        Sets the data model

        :param data: the data to display in the table
        """
        self.path_id = data.select('path_id').unique().item()
        # self.model = TableModel(data.select(['name', 'device', 'path']))
        self.model = EditableTableModel(data, editable_col=self.editable_col)
        self.table_view.setModel(self.model)
        return self


class DataSourceManagement(QDialog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.main_layout = QVBoxLayout(self)

        for grp in pydetecdiv.settings.datapath_list('Test.datapath_list.csv', grouped=True):
            self.main_layout.addWidget(DataSourceGroup(grp[1]))

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Close)
        self.button_box.rejected.connect(self.close)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self.button_box.accepted.connect(self.save_local_datapath)
        self.main_layout.addWidget(self.button_box)

        self.exec()

        self.destroy(True)

    def save_local_datapath(self):
        for child in self.children():
            if isinstance(child, DataSourceGroup):
                print(child, child.model.df.select(['name', 'path_id', 'device', 'MAC', 'path']))


class DataSourceGroup(QGroupBox):
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.main_layout = QVBoxLayout(self)
        first_row = data.filter(polars.col('MAC') == Device.mac())
        if first_row.is_empty():
            first_row = polars.DataFrame({'name': [''],
                                          'path_id': [data['path_id'].unique().item()],
                                          'device': [Device.name()],
                                          'MAC': [Device.mac()],
                                          'path': ['']
                                          })
        print(f'{data["path_id"].unique().item()=} {first_row=}')
        next_rows = data.filter(polars.col('MAC') != Device.mac())
        data = polars.concat([first_row, next_rows])
        self.model = EditableTableModel(data.select(['device', 'name', 'path', 'MAC', 'path_id']), editable_col=[1, 2],
                                        editable_row=[0])

        self.table_view = QTableView(self)
        self.table_view.setModel(self.model)
        self.table_view.setColumnHidden(3, True)
        self.table_view.setColumnHidden(4, True)
        self.horizontal_header = self.table_view.horizontalHeader()
        self.vertical_header = self.table_view.verticalHeader()
        self.horizontal_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.vertical_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontal_header.setStretchLastSection(True)
        self.main_layout.addWidget(self.table_view)
