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
        if self.workspace.text() and self.bioit_conf.text() and os.path.exists(self.bioit_conf.text()):
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

    def select_bioit(self):
        """
        A method opening a File chooser to select the BioImageIT configuration file
        """
        dir_name = os.path.dirname(self.settings.value('bioimageit/config_file'))
        filters = 'JSON files (*.json)'
        if dir_name != os.path.dirname(self.bioit_conf.text()) and self.bioit_conf.text():
            dir_name = self.bioit_conf.text()
        conf_file, _ = QFileDialog.getOpenFileName(self, caption='Choose source files',
                                                dir=dir_name,
                                                filter=filters,
                                                selectedFilter=filters)

        if conf_file  != self.bioit_conf.text():
            self.bioit_conf.setText(conf_file)

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
