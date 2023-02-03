import os.path

from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QLineEdit, QDialogButtonBox, QComboBox, QMessageBox, QDialog,
                               QInputDialog)
from pydetecdiv.domain.Project import Project
from pydetecdiv.persistence.project import list_projects
import pydetecdiv.app
import pydetecdiv.app.gui.resources



class NewProjectDialog(QDialog):
    def __init__(self):
        super().__init__(pydetecdiv.app.main_window)

        self.setWindowModality(Qt.WindowModal)
        self.setWindowTitle('Create project')

        self.layout = QVBoxLayout(self)
        self.label = QLabel('Enter a name for your new project:')
        self.project_name = QLineEdit('MyProject')

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
            self.label.setText(f'Creating {p_name}, please wait.')
            self.project_name.setDisabled(True)
            self.buttonBox.setDisabled(True)
            self.adjustSize()
            self.repaint()
            pydetecdiv.app.project = Project(p_name)
            pydetecdiv.app.main_window.setWindowTitle(f'pyDetecDiv: {p_name}')

            self.hide()


class NewProject(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&New project", parent)
        if icon:
            self.setIcon(QIcon(":icons/new_project"))
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
            self.setIcon(QIcon(":icons/open_project"))
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
