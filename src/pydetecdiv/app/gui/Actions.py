import os.path

from PySide6.QtCore import Slot, Qt, QRect, QFileSelector, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QVBoxLayout, QLineEdit, QDialogButtonBox, \
    QGridLayout, QPushButton, QFileDialog, QComboBox, QErrorMessage, QMessageBox, QDialog, QProgressDialog
from pydetecdiv.domain.Project import Project
from pydetecdiv.persistence.project import list_projects
import pydetecdiv.app

class NewProjectDialog(QWidget):
    def __init__(self):
        super().__init__(pydetecdiv.app.main_window, Qt.Dialog)
        self.project_list = list_projects()
        self.setWindowModality(Qt.WindowModal)

        self.layout = QVBoxLayout(self)
        self.label = QLabel('Enter the name for your new project:')
        self.project_name = QLineEdit()

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.cancel)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.project_name)
        self.layout.addWidget(self.buttonBox)

    def accept(self):
        p_name = self.project_name.text()
        if p_name in list_projects():
            error_msg = QMessageBox(self)
            error_msg.setText(f'Error: {p_name} project already exists!!!')
            error_msg.exec()
        else:
            pydetecdiv.app.project = Project(self.project_name.text())
            pydetecdiv.app.main_window.setWindowTitle(f'pyDetecDiv: {pydetecdiv.app.project.dbname}')
            self.hide()

    def cancel(self):
        self.hide()


class NewProject(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&New project", parent)
        self.w = NewProjectDialog()
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/folder--plus.png"))
        self.triggered.connect(self.w.show)
        parent.addAction(self)


class OpenProjectDialog(QWidget):
    def __init__(self):
        super().__init__(pydetecdiv.app.main_window, Qt.Dialog)
        self.project_list = list_projects()
        self.setWindowModality(Qt.WindowModal)

        self.layout = QVBoxLayout(self)
        self.label = QLabel('Select a project:')
        self.combo = QComboBox()
        self.combo.addItems(self.project_list)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.cancel)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.combo)
        self.layout.addWidget(self.buttonBox)


    def accept(self):
        pydetecdiv.app.project = Project(self.combo.currentText())
        pydetecdiv.app.main_window.setWindowTitle(f'pyDetecDiv: {pydetecdiv.app.project.dbname}')
        self.hide()

    def cancel(self):
        self.hide()

    def refresh(self):
        self.combo.clear()
        self.combo.addItems(list_projects())

    @Slot()
    def show(self) -> None:
        self.refresh()
        super().show()


class OpenProject(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&Open project", parent)
        self.w = OpenProjectDialog()
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/folder-horizontal-open.png"))
            # self.setIcon(QIcon.fromTheme('folder-open'))
        self.triggered.connect(self.w.show)
        parent.addAction(self)


class SettingsDialog(QWidget):
    def __init__(self):
        super().__init__(pydetecdiv.app.main_window, Qt.Dialog)
        self.settings = pydetecdiv.app.get_settings()
        self.setWindowModality(Qt.WindowModal)

        self.file_dialog = QFileDialog(self)
        self.file_dialog.setWindowModality(Qt.WindowModal)

        self.setFixedSize(534, 150)
        self.setObjectName('Settings')
        self.setWindowTitle('Settings')
        self.workspace = QLineEdit()
        self.bioit_conf = QLineEdit()

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
                                          QDialogButtonBox.RestoreDefaults,
                                          Qt.Horizontal)
        self.buttonBox.clicked.connect(self.clicked)
        self.gridLayout.addWidget(self.buttonBox, 4, 0, 1, 2)

    def select_workspace(self):
        dir_name = os.path.dirname(self.settings.value("project/workspace"))
        base_name = os.path.basename(self.settings.value("project/workspace"))
        self.file_dialog.setDirectory(dir_name)
        self.file_dialog.selectFile(base_name)
        self.file_dialog.setFileMode(QFileDialog.Directory)
        self.file_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        self.file_dialog.show()
        self.file_dialog.fileSelected.connect(self.update_workspace)

    def select_bioit(self):
        dir_name = os.path.dirname(self.settings.value("bioimageit/config_file"))
        file_name = os.path.basename(self.settings.value("bioimageit/config_file"))
        self.file_dialog.setDirectory(dir_name)
        self.file_dialog.selectFile(file_name)
        self.file_dialog.setFileMode(QFileDialog.ExistingFile)
        self.file_dialog.setOption(QFileDialog.ShowDirsOnly, False)
        self.file_dialog.setNameFilters(["JSON files (*.json)"])
        self.file_dialog.selectNameFilter("JSON files (*.json)")
        self.file_dialog.show()
        self.file_dialog.fileSelected.connect(self.update_bioit)

    def update_bioit(self):
        self.bioit_conf.setText(self.file_dialog.selectedFiles()[0])

    def update_workspace(self):
        self.workspace.setText(self.file_dialog.selectedFiles()[0])

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

    @Slot()
    def show(self) -> None:
        self.reset()
        super().show()


class Settings(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("Settings", parent)
        self.w = SettingsDialog()
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/gear.png"))
            # self.setIcon(QIcon.fromTheme('preferences-desktop'))
        self.triggered.connect(self.w.show)
        parent.addAction(self)


class Quit(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&Quit", parent)
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/door-open-out.png"))
            # self.setIcon(QIcon.fromTheme('application-exit'))
        self.triggered.connect(self.quit_app)
        parent.addAction(self)

    @Slot()
    def quit_app(self):
        QApplication.quit()
