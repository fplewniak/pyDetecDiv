import polars
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QHeaderView, QVBoxLayout, QSizePolicy, QDialogButtonBox, QTableView, QDialog, QPushButton, QLineEdit,
                               QGroupBox, QHBoxLayout, QFileDialog, QWidget, QLabel)

from pydetecdiv.app.models import TableModel


class TableEditor(QDialog):
    def __init__(self, title=None, **kwargs):
        super().__init__(**kwargs)
        if title is not None:
            self.setWindowTitle(title)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        table_label = QLabel(self)
        table_label.setText('Source paths defined on other devices:')
        self.model = TableModel(polars.DataFrame([{'name': '', 'device': '', 'path': ''}]))
        self.table_view = QTableView(self)
        self.table_view.setModel(self.model)

        # # QTableView Headers
        self.horizontal_header = self.table_view.horizontalHeader()
        self.vertical_header = self.table_view.verticalHeader()
        self.horizontal_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.vertical_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontal_header.setStretchLastSection(True)

        path_group = QGroupBox(self)
        path_group.setTitle('Define corresponding local path:')
        self.path_edit = QLineEdit(path_group)
        button_path = QPushButton(path_group)
        icon = QIcon(":icons/file_chooser")
        button_path.setIcon(icon)
        button_path.clicked.connect(self.select_path)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Close)
        self.button_box.accepted.connect(self.save_local_datapath)
        self.button_box.rejected.connect(self.close)

        self.main_layout = QVBoxLayout(self)
        # table_layout = QHBoxLayout(table_group)
        path_layout = QHBoxLayout(path_group)

        # table_layout.addWidget(self.table_view)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(button_path)

        # self.main_layout.addWidget(table_group)
        self.main_layout.addWidget(table_label)
        self.main_layout.addWidget(self.table_view)
        self.main_layout.addWidget(path_group)
        self.main_layout.addWidget(self.button_box)



    def set_data(self, data: polars.DataFrame):
        self.model = TableModel(data)
        self.table_view.setModel(self.model)

    def select_path(self):
        dir_name = '/'
        if dir_name != self.path_edit.text() and self.path_edit.text():
            dir_name = self.path_edit.text()
        directory = QFileDialog.getExistingDirectory(self, caption='Choose data source directory', dir=dir_name,
                                                     options=QFileDialog.Option.ShowDirsOnly)
        if directory:
            self.path_edit.setText(directory)

    def save_local_datapath(self):
        print(self.path_edit.text())
        self.close()
