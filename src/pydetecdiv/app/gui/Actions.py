import os.path

from PySide6.QtCore import Slot, Qt, QRect, QRegularExpression
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QDialogButtonBox,
    QGridLayout, QPushButton, QFileDialog, QComboBox, QMessageBox, QDialog, QInputDialog)
from pydetecdiv.domain.Project import Project
from pydetecdiv.persistence.project import list_projects
import pydetecdiv.app


class NewProjectDialog(QDialog):
    def __init__(self):
        super().__init__(pydetecdiv.app.main_window)

        self.setWindowModality(Qt.WindowModal)
        self.setWindowTitle('Create project')

        self.layout = QVBoxLayout(self)
        self.label = QLabel('Enter a name for your new project:')
        self.project_name = QLineEdit()

        name_filter = QRegularExpression()
        name_filter.setPattern('\\w[\\w-]*')
        validator = QRegularExpressionValidator()
        validator.setRegularExpression(name_filter)
        self.project_name.setValidator(validator)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.hide)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.project_name)
        self.layout.addWidget(self.buttonBox)

        self.exec()
        self.destroy()

    def accept(self):
        p_name = self.project_name.text()
        if p_name in list_projects():
            error_msg = QMessageBox(self)
            error_msg.setText(f'Error: {p_name} project already exists!!!')
            error_msg.exec()
        elif len(p_name) == 0:
            ...
        else:
            self.setDisabled(True)
            self.repaint()
            pydetecdiv.app.project = Project(self.project_name.text())
            pydetecdiv.app.main_window.setWindowTitle(f'pyDetecDiv: {pydetecdiv.app.project.dbname}')

            self.hide()


class NewProject(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&New project", parent)
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/folder--plus.png"))
        self.triggered.connect(NewProjectDialog)
        parent.addAction(self)

    def create_project(self):
        dialog = QInputDialog()
        dialog.setWindowTitle('Create project')
        dialog.setLabelText('Enter a name for your new project:')
        dialog.setTextValue('MyProject')
        ok = dialog.exec()
        if ok:
            p_name = self.project_name.text()
            if p_name in list_projects():
                error_msg = QMessageBox(self)
                error_msg.setText(f'Error: {p_name} project already exists!!!')
                error_msg.exec()
            else:
                self.setDisabled(True)
                self.repaint()
                pydetecdiv.app.project = Project(self.project_name.text())
                pydetecdiv.app.main_window.setWindowTitle(f'pyDetecDiv: {pydetecdiv.app.project.dbname}')


class OpenProjectDialog(QDialog):
    def __init__(self):
        super().__init__(pydetecdiv.app.main_window)
        self.project_list = list_projects()

        self.setWindowModality(Qt.WindowModal)
        self.setWindowTitle('Open project')

        self.layout = QVBoxLayout(self)
        self.label = QLabel('Select a project name:')
        self.combo = QComboBox()
        self.combo.addItems(self.project_list)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.hide)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.combo)
        self.layout.addWidget(self.buttonBox)

        self.exec()
        self.destroy(True)

    def accept(self):
        pydetecdiv.app.project = Project(self.combo.currentText())
        pydetecdiv.app.main_window.setWindowTitle(f'pyDetecDiv: {pydetecdiv.app.project.dbname}')
        self.hide()

    def refresh(self):
        self.combo.clear()
        self.combo.addItems(list_projects())


class OpenProject(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&Open project", parent)
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/folder-horizontal-open.png"))
            # self.setIcon(QIcon.fromTheme('folder-open'))
        self.triggered.connect(OpenProjectDialog)
        # self.triggered.connect(self.open_project)
        parent.addAction(self)

    def open_project(self):
        dialog = QInputDialog()
        dialog.setWindowTitle('Open project')
        dialog.setLabelText('Select a project name:')
        dialog.setComboBoxItems(sorted(list_projects()))
        dialog.setComboBoxEditable(False)
        ok = dialog.exec()
        if ok:
            pydetecdiv.app.project = Project(dialog.textValue())
            pydetecdiv.app.main_window.setWindowTitle(f'pyDetecDiv: {pydetecdiv.app.project.dbname}')


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
        icon = QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/folder-horizontal.png")
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
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/gear.png"))
            # self.setIcon(QIcon.fromTheme('preferences-desktop'))
        self.triggered.connect(SettingsDialog)
        parent.addAction(self)


class Quit(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&Quit", parent)
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/door-open-out.png"))
            # self.setIcon(QIcon.fromTheme('application-exit'))
        self.triggered.connect(QApplication.quit)
        parent.addAction(self)


class Help(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&Help", parent)
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/question.png"))
        self.triggered.connect(lambda: print(len(QApplication.allWindows())))
        parent.addAction(self)
