"""
Handling actions to edit and manage settings
"""
import os.path

from PySide6.QtCore import Slot, Qt, QRect
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QWidget, QLabel, QLineEdit, QDialogButtonBox, QGridLayout, QPushButton, QFileDialog,
                               QDialog)
from pydetecdiv.app import PyDetecDivApplication, get_settings


class SettingsDialog(QDialog):
    """
    A dialog window to edit settings
    """

    def __init__(self):
        super().__init__(PyDetecDivApplication.main_window)
        self.settings = get_settings()
        self.setWindowModality(Qt.WindowModal)

        self.setFixedSize(534, 150)
        self.setObjectName('Settings')
        self.setWindowTitle('Settings')

        self.workspace = QLineEdit()
        self.bioit_conf = QLineEdit()
        self.reset()

        grid_layout_widget = QWidget(self)
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(5, 5, 5, 5)
        grid_layout_widget.setObjectName("grid_layout_widget")
        grid_layout_widget.setGeometry(QRect(5, 5, 524, 140))
        grid_layout = QGridLayout(grid_layout_widget)

        grid_layout.setContentsMargins(0, 0, 0, 0)

        icon = QIcon(":icons/file_chooser")
        label_workspace = QLabel('Workspace:')
        label_workspace.setAlignment(Qt.AlignBottom)
        button_workspace = QPushButton()
        button_workspace.setIcon(icon)
        button_workspace.clicked.connect(self.select_workspace)

        label_bioit = QLabel('BioImageIT configuration:')
        label_bioit.setAlignment(Qt.AlignBottom)
        button_bioit = QPushButton()
        button_bioit.setIcon(icon)
        button_bioit.clicked.connect(self.select_bioit)

        grid_layout.addWidget(label_workspace, 0, 0, 1, 1)
        grid_layout.addWidget(self.workspace, 1, 0, 1, 1)
        grid_layout.addWidget(button_workspace, 1, 1, 1, 1)

        grid_layout.addWidget(label_bioit, 2, 0, 1, 1)
        grid_layout.addWidget(self.bioit_conf, 3, 0, 1, 1)
        grid_layout.addWidget(button_bioit, 3, 1, 1, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok |
                                           QDialogButtonBox.Cancel |
                                           QDialogButtonBox.Apply |
                                           QDialogButtonBox.Reset,
                                           Qt.Horizontal)
        self.button_box.clicked.connect(self.clicked)
        self.grid_layout.addWidget(self.button_box, 4, 0, 1, 2)

        self.exec()
        self.destroy(True)

    def select_workspace(self):
        """
        Method opening a Directory chooser to select the workspace directory
        """
        dir_name = os.path.dirname(self.settings.value("project/workspace"))
        base_name = os.path.basename(self.settings.value("project/workspace"))
        dir_dialog = QFileDialog(self)
        dir_dialog.setWindowModality(Qt.WindowModal)
        dir_dialog.setDirectory(dir_name)
        dir_dialog.selectFile(base_name)
        dir_dialog.setFileMode(QFileDialog.Directory)
        dir_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dir_dialog.fileSelected.connect(self.workspace.setText)
        dir_dialog.exec()
        dir_dialog.destroy()

    def select_bioit(self):
        """
        A method opening a File chooser to select the BioImageIT configuration file
        """
        dir_name = os.path.dirname(self.settings.value("bioimageit/config_file"))
        file_name = os.path.basename(self.settings.value("bioimageit/config_file"))
        file_dialog = QFileDialog(self)
        file_dialog.setWindowModality(Qt.WindowModal)
        file_dialog.setDirectory(dir_name)
        file_dialog.selectFile(file_name)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setOption(QFileDialog.ShowDirsOnly, False)
        file_dialog.setNameFilters(["JSON files (*.json)"])
        file_dialog.selectNameFilter("JSON files (*.json)")
        file_dialog.fileSelected.connect(self.bioit_conf.setText)
        file_dialog.exec()
        file_dialog.destroy()

    @Slot()
    def clicked(self, button):
        """
        Slot responding to a click on one of the buttons in the button box.
        :param button: the clicked button
        """
        match self.button_box.buttonRole(button):
            case QDialogButtonBox.ButtonRole.ResetRole:
                self.reset()
            case QDialogButtonBox.ButtonRole.ApplyRole:
                self.apply()
            case QDialogButtonBox.ButtonRole.AcceptRole:
                self.apply()
                self.hide()
            case QDialogButtonBox.RejectRole:
                self.hide()

    def apply(self):
        """
        Save the contents of the settings editor into the settings file when the OK button has been clicked
        """
        self.settings.setValue("project/workspace", self.workspace.text())
        self.settings.setValue("bioimageit/config_file", self.bioit_conf.text())

    def reset(self):
        """
        Reset the contents of the settings editor to the values currently in the settings file
        """
        self.workspace.setText(self.settings.value("project/workspace"))
        self.bioit_conf.setText(self.settings.value("bioimageit/config_file"))


class Settings(QAction):
    """
    Action to open a session editor window
    """

    def __init__(self, parent, icon=False):
        super().__init__(QIcon(":icons/settings"), "Settings", parent)
        self.triggered.connect(SettingsDialog)
        parent.addAction(self)
