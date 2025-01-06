"""
Handling actions to edit and manage settings
"""
import getpass
import os.path

from PySide6.QtCore import Slot, Qt, QDir
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QLineEdit, QDialogButtonBox, QPushButton, QFileDialog, QDialog, QHBoxLayout, QVBoxLayout,
                               QGroupBox, QComboBox, QSpinBox, QWidget)
from pydetecdiv.app import PyDetecDiv, get_settings
from pydetecdiv.persistence import implemented_dbms


class SettingsDialog(QDialog):
    """
    A dialog window to edit application settings
    """

    def __init__(self):
        super().__init__(PyDetecDiv.main_window)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setObjectName('Settings')
        self.setWindowTitle('Settings')

        self.settings = get_settings()

        # Widgets
        conda_group = QGroupBox(self)
        conda_group.setTitle('Conda path')
        self.conda_path = QLineEdit(conda_group)
        icon = QIcon(":icons/file_chooser")
        button_conda = QPushButton(conda_group)
        button_conda.setIcon(icon)

        batch_group = QGroupBox(self)
        batch_group.setTitle('Batch size')
        self.batch_size = QSpinBox(self)
        self.batch_size.setRange(256, 16384)
        self.batch_size.setSingleStep(256)

        appdata_group = QGroupBox(self)
        appdata_group.setTitle('Application data:')
        self.appdata = QLineEdit(appdata_group)
        button_appdata = QPushButton(appdata_group)
        button_appdata.setIcon(icon)

        tools_group = QGroupBox(self)
        tools_group.setTitle('Toolbox:')
        self.toolbox = QLineEdit(tools_group)
        button_toolbox = QPushButton(tools_group)
        button_toolbox.setIcon(icon)

        user_group = QGroupBox(self)
        user_group.setTitle('User:')
        self.user = QLineEdit(user_group)

        dbms_group = QGroupBox(self)
        dbms_group.setTitle('DBMS:')
        self.dbms = QComboBox()
        self.dbms.addItems(implemented_dbms())

        workspace_group = QGroupBox(self)
        workspace_group.setTitle('Workspace:')
        self.workspace = QLineEdit(workspace_group)
        button_workspace = QPushButton(workspace_group)
        button_workspace.setIcon(icon)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                           QDialogButtonBox.StandardButton.Cancel |
                                           QDialogButtonBox.StandardButton.Apply |
                                           QDialogButtonBox.StandardButton.Reset,
                                           Qt.Orientation.Horizontal)

        # Layout
        vertical_layout = QVBoxLayout(self)
        workspace_layout = QHBoxLayout(workspace_group)
        dbms_layout = QHBoxLayout(dbms_group)
        user_layout = QHBoxLayout(user_group)
        appdata_layout = QHBoxLayout(appdata_group)
        tools_layout = QHBoxLayout(tools_group)
        conda_layout = QHBoxLayout(conda_group)
        batch_layout = QHBoxLayout(batch_group)

        workspace_layout.addWidget(self.workspace)
        workspace_layout.addWidget(button_workspace)

        appdata_layout.addWidget(self.appdata)
        appdata_layout.addWidget(button_appdata)

        tools_layout.addWidget(self.toolbox)
        tools_layout.addWidget(button_toolbox)

        conda_layout.addWidget(self.conda_path)
        conda_layout.addWidget(button_conda)

        batch_layout.addWidget(self.batch_size)

        dbms_layout.addWidget(self.dbms)

        user_layout.addWidget(self.user)

        vertical_layout.addWidget(workspace_group)
        vertical_layout.addWidget(appdata_group)
        vertical_layout.addWidget(tools_group)
        vertical_layout.addWidget(conda_group)
        vertical_layout.addWidget(user_group)
        vertical_layout.addWidget(batch_group)
        vertical_layout.addWidget(dbms_group)

        vertical_layout.addWidget(self.button_box)

        # Widget behaviour
        button_workspace.clicked.connect(self.select_workspace)
        button_appdata.clicked.connect(self.select_appdata_path)
        button_toolbox.clicked.connect(self.select_toolbox_path)
        self.button_box.clicked.connect(self.clicked)
        self.workspace.textChanged.connect(self.toggle_buttons)
        self.user.textChanged.connect(self.toggle_buttons)
        self.toolbox.textChanged.connect(self.toggle_buttons)

        self.reset()
        self.exec()
        self.destroy(True)

    def toggle_buttons(self) -> None:
        """
        Enable or disable OK and Apply buttons depending upon the validity of input text
        """
        if self.workspace.text() and self.user.text():
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
            self.button_box.button(QDialogButtonBox.StandardButton.Apply).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            self.button_box.button(QDialogButtonBox.StandardButton.Apply).setEnabled(False)

    def select_workspace(self) -> None:
        """
        Method opening a Directory chooser to select the workspace directory
        """
        dir_name = str(os.path.join(os.path.dirname(str(self.settings.value("project/workspace"))),
                                    os.path.basename(str(self.settings.value("project/workspace")))))
        if dir_name != self.workspace.text() and self.workspace.text():
            dir_name = self.workspace.text()
        directory = QFileDialog.getExistingDirectory(self, caption='Choose workspace directory', dir=dir_name,
                                                     options=QFileDialog.Option.ShowDirsOnly)
        if directory:
            self.workspace.setText(directory)

    def select_appdata_path(self) -> None:
        """
        Method opening a Directory chooser to select the toolbox directory
        """
        dir_name = str(os.path.join(os.path.dirname(str(self.settings.value("paths/appdata"))),
                                    os.path.basename(str(self.settings.value("paths/appdata")))))
        if dir_name != self.appdata.text() and self.appdata.text():
            dir_name = self.appdata.text()
        directory = QFileDialog.getExistingDirectory(self, caption='Choose application data directory', dir=dir_name,
                                                     options=QFileDialog.Option.ShowDirsOnly)
        if directory:
            self.appdata.setText(directory)

    def select_toolbox_path(self) -> None:
        """
        Method opening a Directory chooser to select the toolbox directory
        """
        dir_name = str(os.path.join(os.path.dirname(str(self.settings.value("paths/toolbox"))),
                                    os.path.basename(str(self.settings.value("paths/toolbox")))))
        if dir_name != self.toolbox.text() and self.toolbox.text():
            dir_name = self.toolbox.text()
        directory = QFileDialog.getExistingDirectory(self, caption='Choose toolbox directory', dir=dir_name,
                                                     options=QFileDialog.Option.ShowDirsOnly)
        if directory:
            self.toolbox.setText(directory)

    @Slot()
    def clicked(self, button: QPushButton):
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
            case QDialogButtonBox.ButtonRole.RejectRole:
                self.hide()

    def apply(self):
        """
        Save the contents of the settings editor into the settings file when the OK button has been clicked
        """

        self.settings.setValue("project/workspace", self.workspace.text())
        QDir().mkpath(self.workspace.text())
        self.settings.setValue("paths/appdata", self.appdata.text())
        # self.settings.setValue("paths/toolbox", self.toolbox.text())
        self.settings.setValue("project.conda/dir", self.conda_path.text())
        self.settings.setValue("project/user", self.user.text())
        self.settings.setValue("project/dbms", implemented_dbms()[self.dbms.currentIndex()])
        self.settings.setValue("project/batch", self.batch_size.value())

    def reset(self):
        """
        Reset the contents of the settings editor to the values currently in the settings file
        """
        self.workspace.setText(str(self.settings.value("project/workspace")))
        self.appdata.setText(str(self.settings.value("paths/appdata")))
        self.toolbox.setText(str(self.settings.value("paths/toolbox")))
        self.conda_path.setText(str(self.settings.value("project.conda/dir")))
        self.user.setText(str(self.settings.value("project/user")))
        self.dbms.setCurrentIndex(implemented_dbms().index(self.settings.value("project/dbms")))
        self.batch_size.setValue(int(str(self.settings.value("project/batch"))))


class Settings(QAction):
    """
    Action to open a session editor window
    """

    def __init__(self, parent: QWidget):
        super().__init__(QIcon(":icons/settings"), "Settings", parent)
        self.triggered.connect(SettingsDialog)
        parent.addAction(self)
