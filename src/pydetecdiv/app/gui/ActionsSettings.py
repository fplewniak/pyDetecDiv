"""
Handling actions to edit and manage settings
"""
import os.path

from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QLineEdit, QDialogButtonBox, QPushButton, QFileDialog, QDialog, QHBoxLayout, QVBoxLayout,
                               QGroupBox)
from pydetecdiv.app import PyDetecDivApplication, get_settings


class SettingsDialog(QDialog):
    """
    A dialog window to edit applicatino settings
    """

    def __init__(self):
        super().__init__(PyDetecDivApplication.main_window)
        self.setWindowModality(Qt.WindowModal)
        self.setObjectName('Settings')
        self.setWindowTitle('Settings')

        self.settings = get_settings()

        # Widgets
        workspace_group = QGroupBox(self)
        workspace_group.setTitle('Workspace:')
        self.workspace = QLineEdit(workspace_group)
        icon = QIcon(":icons/file_chooser")
        button_workspace = QPushButton(workspace_group)
        button_workspace.setIcon(icon)

        bioit_group = QGroupBox(self)
        bioit_group.setTitle('BioImageIT configuration:')
        self.bioit_conf = QLineEdit(bioit_group)
        button_bioit = QPushButton(bioit_group)
        button_bioit.setIcon(icon)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok |
                                           QDialogButtonBox.Cancel |
                                           QDialogButtonBox.Apply |
                                           QDialogButtonBox.Reset,
                                           Qt.Horizontal)

        # Layout
        vertical_layout = QVBoxLayout(self)
        workspace_layout = QHBoxLayout(workspace_group)

        workspace_layout.addWidget(self.workspace)
        workspace_layout.addWidget(button_workspace)

        bioit_layout = QHBoxLayout(bioit_group)
        bioit_layout.addWidget(self.bioit_conf)
        bioit_layout.addWidget(button_bioit)

        vertical_layout.addWidget(workspace_group)
        vertical_layout.addWidget(bioit_group)
        vertical_layout.addWidget(self.button_box)

        # Widget behaviour
        button_workspace.clicked.connect(self.select_workspace)
        button_bioit.clicked.connect(self.select_bioit)
        self.button_box.clicked.connect(self.clicked)
        self.workspace.textChanged.connect(self.toggle_buttons)
        self.bioit_conf.textChanged.connect(self.toggle_buttons)

        self.reset()
        self.exec()
        self.destroy(True)

    def toggle_buttons(self):
        """
        Enable or disable OK and Apply buttons depending upon the validity of input text
        """
        if self.workspace.text() and self.bioit_conf.text():
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
            self.button_box.button(QDialogButtonBox.Apply).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            self.button_box.button(QDialogButtonBox.Apply).setEnabled(False)

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

    def __init__(self, parent):
        super().__init__(QIcon(":icons/settings"), "Settings", parent)
        self.triggered.connect(SettingsDialog)
        parent.addAction(self)
