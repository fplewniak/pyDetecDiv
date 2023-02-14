"""
Handling actions to open, create and interact with projects
"""
import glob, os

from PySide6.QtCore import Qt, QRegularExpression, QStringListModel
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QFileDialog, QDialog, QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QDialogButtonBox, QListView, QComboBox)
from pydetecdiv.app import PyDetecDivApplication, get_settings, WaitDialog, PyDetecDivThread, pydetecdiv_project


class ImportDataDialog(QDialog):
    def __init__(self):
        super().__init__(PyDetecDivApplication.main_window)
        settings = get_settings()
        self.project_path = os.path.join(settings.value("project/workspace"), PyDetecDivApplication.project_name)
        self.setWindowModality(Qt.WindowModal)

        self.setObjectName('ImportData')
        self.setWindowTitle('Import image data')

        # Widgets
        self.source_group_box = QGroupBox(self)
        self.source_group_box.setTitle('Source for image files to import:')

        self.buttons_widget = QWidget(self.source_group_box)
        self.files_button = QPushButton('Add files', self.buttons_widget)
        self.directory_button = QPushButton('Add directory', self.buttons_widget)
        self.path_button = QPushButton('Add path', self.buttons_widget)

        self.list_view = QListView(self.source_group_box)
        self.selection_model = QStringListModel()
        self.list_view.setModel(self.selection_model)

        self.destination_widget = QGroupBox(self)
        self.destination_widget.setTitle(f'Destination: {self.project_path}/data/')
        self.destination_directory = QComboBox(self.destination_widget)
        self.destination_directory.addItems(self.get_destinations())
        self.destination_directory.setEditable(True)
        self.destination_directory.setValidator(self.sub_directory_name_validator())

        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok, self)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

        # Layout
        self.vertical_layout = QVBoxLayout(self)
        self.source_layout = QVBoxLayout(self.source_group_box)
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        self.destination_layout = QHBoxLayout(self.destination_widget)

        self.source_layout.addWidget(self.list_view)
        self.source_layout.addWidget(self.buttons_widget)

        self.buttons_layout.addWidget(self.files_button)
        self.buttons_layout.addWidget(self.directory_button)
        self.buttons_layout.addWidget(self.path_button)

        self.destination_layout.addWidget(self.destination_directory)

        self.vertical_layout.addWidget(self.source_group_box)
        self.vertical_layout.addWidget(self.destination_widget)
        self.vertical_layout.addWidget(self.button_box)

        # Widgets behaviour
        #
        self.files_button.clicked.connect(
            lambda _: AddFilesDialog(self, selection=self.selection_model, choose_dir=False))
        self.directory_button.clicked.connect(
            lambda _: AddFilesDialog(self, selection=self.selection_model, choose_dir=True))
        self.path_button.clicked.connect(lambda _: AddPathDialog(self, selection=self.selection_model))

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)

        self.selection_model.dataChanged.connect(self.selection_is_not_empty)

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy(True)

    def get_destinations(self):
        return [''] + [d.name for d in os.scandir(os.path.join(self.project_path, 'data')) if d.is_dir()]

    def accept(self):
        file_list = self.selection_model.stringList()
        print(f'Importing files {file_list}')
        self.close()

    def selection_is_not_empty(self):
        print(self.selection_model.stringList())
        if self.selection_model.stringList():
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

    @staticmethod
    def sub_directory_name_validator():
        """
        Name validator to filter invalid character in directory name
        :return: the validator
        :rtype: QRegularExpressionValidator
        """
        name_filter = QRegularExpression()
        name_filter.setPattern('\\w[\\w-]*')
        validator = QRegularExpressionValidator()
        validator.setRegularExpression(name_filter)
        return validator


class ImportData(QAction):
    """
    Action to import raw data images into a project
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/import_images"), "&Import image files", parent)
        self.triggered.connect(ImportDataDialog)
        self.setEnabled(False)
        parent.addAction(self)


class AddFilesDialog(QFileDialog):
    """
    A dialog window to select files or directory to add to import list
    """

    def __init__(self, parent_window, selection, choose_dir=False):
        super().__init__(parent_window)
        self.setWindowModality(Qt.WindowModal)
        self.selection = selection

        if choose_dir:
            self.setFileMode(QFileDialog.Directory)
        else:
            self.setFileMode(QFileDialog.ExistingFiles)
            self.setNameFilters(["TIFF (*.tif *.tiff)",
                                 "JPEG (*.jpg *.jpeg)",
                                 "PNG (*.png)",
                                 "Image files (*.tif *.tiff, *.jpg *.jpeg, *.png)", ])

        self.filesSelected.connect(self.select_files)

        self.exec()
        self.destroy(True)

    def select_files(self):
        file_selection = self.selection.stringList()
        self.selection.setStringList(file_selection + self.selectedFiles())
        self.parent().button_box.button(QDialogButtonBox.Ok).setEnabled(True)


class AddPathDialog(QDialog):
    """
    A dialog window to select a path pointing to files or directories to import
    """

    def __init__(self, parent_window, selection):
        super().__init__(parent_window)
        self.setWindowModality(Qt.WindowModal)
        self.selection = selection

        self.path_widget = QWidget(self)
        self.path_label = QLabel('Path:', self.path_widget)
        self.path_text_input = QLineEdit(self.path_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                           Qt.Horizontal, self.path_widget)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self.button_box.button(QDialogButtonBox.Apply).setEnabled(False)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self.add_path)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.path_widget)
        self.layout.addWidget(self.button_box)
        self.path_layout = QHBoxLayout(self.path_widget)
        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path_text_input)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)
        self.path_text_input.textChanged.connect(self.path_specification_changed)

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy(True)

    def accept(self):
        self.add_path()
        self.close()

    def add_path(self):
        file_selection = self.selection.stringList()
        self.selection.setStringList(file_selection + [self.path_text_input.text()])
        self.parent().button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def path_specification_changed(self):
        if os.path.exists(self.path_text_input.text()):
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
            self.button_box.button(QDialogButtonBox.Apply).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            self.button_box.button(QDialogButtonBox.Apply).setEnabled(False)
