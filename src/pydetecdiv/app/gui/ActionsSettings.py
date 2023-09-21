"""
Handling actions to edit and manage settings
"""
import os.path

from PySide6.QtCore import Slot, Qt, QDir
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QLineEdit, QDialogButtonBox, QPushButton, QFileDialog, QDialog, QHBoxLayout, QVBoxLayout,
                               QGroupBox)
from pydetecdiv.app import PyDetecDiv, get_settings


class SettingsDialog(QDialog):
    """
    A dialog window to edit application settings
    """

    def __init__(self):
        super().__init__(PyDetecDiv().main_window)
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

        vertical_layout.addWidget(workspace_group)
        # vertical_layout.addWidget(bioit_group)
        vertical_layout.addWidget(self.button_box)

        # Widget behaviour
        button_workspace.clicked.connect(self.select_workspace)
        self.button_box.clicked.connect(self.clicked)
        self.workspace.textChanged.connect(self.toggle_buttons)

        self.reset()
        self.exec()
        self.destroy(True)

    def toggle_buttons(self):
        """
        Enable or disable OK and Apply buttons depending upon the validity of input text
        """
        if self.workspace.text():
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
            self.button_box.button(QDialogButtonBox.Apply).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            self.button_box.button(QDialogButtonBox.Apply).setEnabled(False)

    def select_workspace(self):
        """
        Method opening a Directory chooser to select the workspace directory
        """
        dir_name = str(os.path.join(os.path.dirname(self.settings.value("project/workspace")),
                                os.path.basename(self.settings.value("project/workspace"))))
        if dir_name != self.workspace.text() and self.workspace.text():
            dir_name = self.workspace.text()
        directory = QFileDialog.getExistingDirectory(self, caption='Choose workspace directory', dir=dir_name,
                                                     options=QFileDialog.ShowDirsOnly)
        if directory:
            self.workspace.setText(directory)

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
        QDir().mkpath(self.workspace.text())

    def reset(self):
        """
        Reset the contents of the settings editor to the values currently in the settings file
        """
        self.workspace.setText(self.settings.value("project/workspace"))


class Settings(QAction):
    """
    Action to open a session editor window
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/settings"), "Settings", parent)
        self.triggered.connect(SettingsDialog)
        parent.addAction(self)
