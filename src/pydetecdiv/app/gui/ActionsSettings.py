import os.path

from PySide6.QtCore import Slot, Qt, QRect
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QWidget, QLabel, QLineEdit, QDialogButtonBox, QGridLayout, QPushButton, QFileDialog,
                               QDialog)
import pydetecdiv.app
import pydetecdiv.app.gui.resources


class SettingsDialog(QDialog):
    def __init__(self):
        super().__init__(pydetecdiv.app.main_window)
        self.settings = pydetecdiv.app.get_settings()
        self.setWindowModality(Qt.WindowModal)

        self.setFixedSize(534, 150)
        self.setObjectName('Settings')
        self.setWindowTitle('Settings')
        self.workspace = QLineEdit()
        self.bioit_conf = QLineEdit()
        self.reset()

        self.gridLayoutWidget = QWidget(self)
        self.gridLayout = QGridLayout()
        self.gridLayout.setContentsMargins(5, 5, 5, 5)
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(5, 5, 524, 140))
        self.gridLayout = QGridLayout(self.gridLayoutWidget)

        self.gridLayout.setContentsMargins(0, 0, 0, 0)

        # icon = QIcon(QIcon.fromTheme(u"folder"))
        icon = QIcon(":icons/file_chooser")
        self.label_workspace = QLabel('Workspace:')
        self.label_workspace.setAlignment(Qt.AlignBottom)
        self.button_workspace = QPushButton()
        self.button_workspace.setIcon(icon)
        self.button_workspace.clicked.connect(self.select_workspace)

        self.label_bioit = QLabel('BioImageIT configuration:')
        self.label_bioit.setAlignment(Qt.AlignBottom)
        self.button_bioit = QPushButton()
        self.button_bioit.setIcon(icon)
        self.button_bioit.clicked.connect(self.select_bioit)

        self.gridLayout.addWidget(self.label_workspace, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.workspace, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.button_workspace, 1, 1, 1, 1)

        self.gridLayout.addWidget(self.label_bioit, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.bioit_conf, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.button_bioit, 3, 1, 1, 1)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok |
                                          QDialogButtonBox.Cancel |
                                          QDialogButtonBox.Apply |
                                          QDialogButtonBox.Reset,
                                          Qt.Horizontal)
        self.buttonBox.clicked.connect(self.clicked)
        self.gridLayout.addWidget(self.buttonBox, 4, 0, 1, 2)

        self.exec()
        self.destroy(True)

    def select_workspace(self):
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
    def clicked(self, button, *args, **kwargs):
        match self.buttonBox.buttonRole(button):
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
        self.settings.setValue("project/workspace", self.workspace.text())
        self.settings.setValue("bioimageit/config_file", self.bioit_conf.text())

    def reset(self):
        self.workspace.setText(self.settings.value("project/workspace"))
        self.bioit_conf.setText(self.settings.value("bioimageit/config_file"))


class Settings(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("Settings", parent)
        if icon:
            self.setIcon(QIcon(":icons/settings"))
        self.triggered.connect(SettingsDialog)
        parent.addAction(self)
