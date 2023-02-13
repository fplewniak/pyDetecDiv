"""
Handling actions to open, create and interact with projects
"""
import glob, os

from PySide6.QtCore import Qt, QRect, QRegularExpression
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

        self.path_widget = QWidget(self.source_group_box)
        self.path_label = QLabel('Path:', self.path_widget)
        self.path_text_input = QLineEdit(self.path_widget)

        self.buttons_widget = QWidget(self.source_group_box)
        self.files_button = QPushButton('Choose files', self.buttons_widget)

        self.directory_button = QPushButton('Choose directory', self.buttons_widget)

        self.source_button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Reset, self.buttons_widget)
        self.source_button_box.button(QDialogButtonBox.Apply).setEnabled(False)

        self.list_view = QListView(self.source_group_box)

        self.destination_widget = QGroupBox(self)
        self.destination_widget.setTitle('Destination:')
        self.destination_directory = QComboBox(self.destination_widget)
        self.destination_directory.addItems(self.get_destinations())
        self.destination_directory.setEditable(True)
        self.destination_directory.setValidator(self.sub_directory_name_validator())

        self.dialog_button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok, self)

        # Layout
        self.vertical_layout = QVBoxLayout(self)
        self.source_layout = QVBoxLayout(self.source_group_box)
        self.path_layout = QHBoxLayout(self.path_widget)
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        self.destination_layout = QHBoxLayout(self.destination_widget)

        self.source_layout.addWidget(self.path_widget)
        self.source_layout.addWidget(self.buttons_widget)
        self.source_layout.addWidget(self.list_view)

        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path_text_input)

        self.buttons_layout.addWidget(self.files_button)
        self.buttons_layout.addWidget(self.directory_button)
        self.buttons_layout.addWidget(self.source_button_box)

        self.destination_layout.addWidget(self.destination_directory)

        self.vertical_layout.addWidget(self.source_group_box)
        self.vertical_layout.addWidget(self.destination_widget)
        self.vertical_layout.addWidget(self.dialog_button_box)

        # Widgets behaviour
        self.path_text_input.textChanged.connect(self.path_specification_changed)
        self.files_button.clicked.connect(lambda _: ChooseDialog(choose_dir=False))
        self.directory_button.clicked.connect(lambda _: ChooseDialog(choose_dir=True))

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy(True)

    def path_specification_changed(self):
        if self.path_text_input.text():
            self.source_button_box.button(QDialogButtonBox.Apply).setEnabled(True)
        else:
            self.source_button_box.button(QDialogButtonBox.Apply).setEnabled(False)

    def get_destinations(self):
        return [''] + [d.name for d in os.scandir(os.path.join(self.project_path, 'data')) if d.is_dir()]

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


class ChooseDialog(QFileDialog):
    """
    A dialog window to edit settings
    """

    def __init__(self, choose_dir=False):
        super().__init__(PyDetecDivApplication.main_window)
        self.settings = get_settings()
        self.setWindowModality(Qt.WindowModal)

        if choose_dir:
            self.setFileMode(QFileDialog.Directory)
        else:
            self.setFileMode(QFileDialog.ExistingFiles)

        self.setNameFilters(["TIFF (*.tif *.tiff)",
                             "JPEG (*.jpg *.jpeg)",
                             "PNG (*.png)",
                             "Image files (*.tif *.tiff, *.jpg *.jpeg, *.png)",])

        self.filesSelected.connect(self.import_files)

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy()

    def import_files(self):
        with pydetecdiv_project(PyDetecDivApplication.project_name) as project:
            print(self.selectedFiles())
            # project.import_images(f'{self.selectedFiles()[0]}/*')


class ChooseData(QAction):
    def __init__(self, icon, label, parent, choose_dir=False):
        super().__init__(icon, label, parent)
        self.triggered.connect(lambda _: ChooseDialog(choose_dir=False))
        self.setEnabled(False)
        parent.addAction(self)


class ChooseDataFiles(ChooseData):
    """
    Action to import raw data images into a project
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/import_images"), "&Import image files", parent, choose_dir=False)

class ChooseDataDir(ChooseData):
    """
    Action to import raw data images into a project
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/import_folder"), "Import &directory", parent, choose_dir=True)
