import os
from typing import Self

import polars
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QHeaderView, QVBoxLayout, QSizePolicy, QDialogButtonBox, QTableView, QDialog, QPushButton, QLineEdit,
                               QGroupBox, QHBoxLayout, QFileDialog, QLabel, QTabWidget)

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
        path_layout = QHBoxLayout(path_group)

        path_layout.addWidget(name_label)
        path_layout.addWidget(self.name_edit)
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(button_path)

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


class DataSourceManagementTab(QTabWidget):
    def __init__(self):
        super().__init__()
        self.tab_list = set()
        for grp in pydetecdiv.settings.datapath_list(grouped=True):
            self.add_tab(grp[1])
            # name_df = grp[1].filter(polars.col('MAC') == Device.mac()).select('name')
            # tab_widget = DataSourceGroup(grp[1], self)
            # if name_df.is_empty():
            #     self.addTab(tab_widget, QIcon(":icons/question-button"), '')
            # else:
            #     self.addTab(tab_widget, name_df.item())
            # self.tab_list.add(tab_widget)

    def add_tab(self, data):
        name_df = data.filter(polars.col('MAC') == Device.mac()).select('name')
        tab_widget = DataSourceGroup(data, parent=self)
        if name_df.is_empty() or name_df.item(0, 0).replace(' ', '') == '':
            self.addTab(tab_widget, QIcon(":icons/question-button"), '')
        else:
            self.addTab(tab_widget, name_df.item())
        self.tab_list.add(tab_widget)
        self.setCurrentWidget(tab_widget)

    def path_edit_changed(self):
        if self.currentWidget().name_edit.text().replace(' ', '') == '':
            self.setTabIcon(self.currentIndex(), QIcon(":icons/question-button"))
            self.setTabText(self.currentIndex(), '')
        else:
            self.setTabIcon(self.currentIndex(), QIcon())
            self.setTabText(self.currentIndex(), self.currentWidget().name_edit.text())


class DataSourceManagement(QDialog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.main_layout = QVBoxLayout(self)
        self.tabs = DataSourceManagementTab()
        self.main_layout.addWidget(self.tabs)

        # self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Close)
        self.button_box = QDialogButtonBox()
        self.button_add = self.button_box.addButton('Add source', QDialogButtonBox.ButtonRole.ActionRole)
        self.button_add.clicked.connect(self.add_source)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Close)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.rejected.connect(self.close)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self.button_box.accepted.connect(self.save_local_datapath)
        self.main_layout.addWidget(self.button_box)

        self.exec()

        self.destroy(True)

    def add_source(self):
        path_id = pydetecdiv.settings.create_path_id(as_string=True)
        data = polars.DataFrame({'name'   : [''],
                                 'path_id': [path_id],
                                 'device' : [Device.name()],
                                 'MAC'    : [Device.mac()],
                                 'path'   : ['']
                                 }, schema={'name'   : str,
                                            'path_id': str,
                                            'device' : str,
                                            'MAC'    : str,
                                            'path'   : str
                                            })
        self.tabs.add_tab(data)
        # tab_widget = DataSourceGroup(data, self.tabs)
        # self.tabs.addTab(tab_widget, QIcon(":icons/question-button"), '')
        # self.tabs.tab_list.add(tab_widget)
        # self.tabs.setCurrentWidget(tab_widget)

    def save_local_datapath(self):
        self.data.write_csv(datapath_file())
        self.close()

    @property
    def data(self):
        data = polars.DataFrame(schema={
            'name'   : str,
            'path_id': str,
            'device' : str,
            'MAC'    : str,
            'path'   : str
            })
        for tab in self.tabs.tab_list:
            data.extend(tab.data.select(['name', 'path_id', 'device', 'MAC', 'path']))
        return data


class DataSourceGroup(QGroupBox):
    def __init__(self, data, parent=None, **kwargs):
        super().__init__(**kwargs)
        self._parent = parent

        self.main_layout = QVBoxLayout(self)
        self.this_device = data.filter(polars.col('MAC') == Device.mac())
        if self.this_device.is_empty():
            self.this_device = polars.DataFrame({'name'   : [''],
                                                 'path_id': [data['path_id'].unique().item()],
                                                 'device' : [Device.name()],
                                                 'MAC'    : [Device.mac()],
                                                 'path'   : ['']
                                                 }, schema={'name'   : str,
                                                            'path_id': str,
                                                            'device' : str,
                                                            'MAC'    : str,
                                                            'path'   : str
                                                            })

        path_group_label = QLabel(self)
        path_group_label.setText(f"<p><b>Data source uid: {data['path_id'].unique().item()}</b></p>")
        path_group = QGroupBox(self)
        name_label = QLabel('name')
        self.name_edit = QLineEdit(path_group)
        self.name_edit.setText(self.this_device['name'].item())
        self.name_edit.setClearButtonEnabled(True)
        path_label = QLabel('path')
        self.path_edit = QLineEdit(path_group)
        self.path_edit.setText(self.this_device['path'].item())
        self.path_edit.setClearButtonEnabled(True)
        button_path = QPushButton(path_group)
        button_path.setIcon(QIcon(":icons/file_chooser"))
        button_path.clicked.connect(self.select_path)
        clear_button = QPushButton(path_group)
        clear_button.setIcon(QIcon(":icons/cross-button"))
        clear_button.clicked.connect(self.clear_path)
        self.path_edit.textChanged.connect(self.path_edit_changed)
        self.name_edit.textChanged.connect(self.path_edit_changed)

        path_layout = QHBoxLayout(path_group)
        # table_layout.addWidget(self.table_view)
        path_layout.addWidget(name_label)
        path_layout.addWidget(self.name_edit)
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(button_path)
        path_layout.addWidget(clear_button)

        self.main_layout.addWidget(path_group_label)
        self.main_layout.addWidget(path_group)

        self.other_devices = data.filter(polars.col('MAC') != Device.mac())

        self.other_devices_model = TableModel(self.other_devices.select(['device', 'name', 'path', 'MAC', 'path_id']))
        self.other_devices_view = QTableView(self)
        self.other_devices_view.setModel(self.other_devices_model)
        self.other_devices_view.setColumnHidden(3, True)
        self.other_devices_view.setColumnHidden(4, True)
        self.other_devices_horizontal_header = self.other_devices_view.horizontalHeader()
        self.other_devices_horizontal_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.other_devices_horizontal_header.setStretchLastSection(True)
        self.other_devices_view.verticalHeader().setStretchLastSection(False)
        self.other_devices_view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
        self.other_devices_view.adjustSize()

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
        self.main_layout.addWidget(QLabel('<br/><b>Data source specifications on other devices:</b>'))
        self.main_layout.addWidget(self.other_devices_view)

        self.adjustSize()

    def parent(self):
        return self._parent

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
        Update path and name in the data model when it has been changed in the GUI
        """
        if os.path.isdir(self.path_edit.text()) and self.name_edit.text():
            self.this_device = self.this_device.with_columns(path=polars.lit(self.path_edit.text()),
                                                             name=polars.lit(self.name_edit.text()))
        self.parent().path_edit_changed()

    def clear_path(self) -> None:
        """
        Update path and name in the data model when it has been changed in the GUI
        """
        self.path_edit.clear()
        self.name_edit.clear()
        self.this_device = self.this_device.with_columns(path=polars.lit(self.path_edit.text()),
                                                         name=polars.lit(self.name_edit.text()))

    @property
    def data(self):
        return polars.concat(
                [self.this_device.filter((polars.col('path_id') != '') & (polars.col('name') != '')), self.other_devices])
