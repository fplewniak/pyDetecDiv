"""
Handling actions to open, create and interact with projects
"""
from enum import Enum

from PySide6.QtCore import Qt, QRegularExpression, Slot, Signal
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QLineEdit, QDialogButtonBox, QComboBox, QMessageBox, QDialog,)
from pydetecdiv.app import PyDetecDiv, project_list, WaitDialog, pydetecdiv_project
from pydetecdiv.app import MessageDialog
from pydetecdiv.persistence.project import delete_project
from pydetecdiv.exceptions import OpenProjectError, UnknownRepositoryTypeError

class ProjectAction(Enum):
    New = 1
    Open = 2
    Delete = 3

class ProjectDialog(QDialog):
    """
    A generic dialog window to create or open a project
    """
    finished = Signal(bool, name='projectOpen')

    def __init__(self, project_list, project_action=ProjectAction.Open):
        super().__init__(PyDetecDiv().main_window)
        self.wait = None
        self.project_action = project_action
        self.setWindowModality(Qt.WindowModal)
        self.layout = QVBoxLayout(self)

        match self.project_action:
            case ProjectAction.New:
                self.setWindowTitle('Create project')
                self.label = QLabel('Enter a name for your new project:')
                self.project_name = QLineEdit('MyProject')
                self.project_name.textChanged.connect(self.project_name_changed)
            case ProjectAction.Open | ProjectAction.Delete:
                self.label = QLabel('Select a project name:')
                self.project_name = QComboBox()
                self.project_name.addItems(sorted(project_list))
                match self.project_action:
                    case ProjectAction.Open:
                        self.setWindowTitle('Open project')
                        self.project_name.setEditable(True)
                        self.project_name.editTextChanged.connect(self.project_name_changed)
                    case ProjectAction.Delete:
                        self.setWindowTitle('Delete project')

        self.project_name.setValidator(self.project_name_validator())

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.hide)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.project_name)
        self.layout.addWidget(self.button_box)

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy()

    @staticmethod
    def project_name_validator():
        """
        Name validator to filter invalid character in project name

        :return: the validator
        :rtype: QRegularExpressionValidator
        """
        name_filter = QRegularExpression()
        name_filter.setPattern('\\w[\\w-]*')
        validator = QRegularExpressionValidator()
        validator.setRegularExpression(name_filter)
        return validator

    @Slot()
    def project_name_changed(self):
        """
        Slot checking whether the project name input is empty or not and enabling or disabling Ok button accordingly
        """
        if self.get_project_name():
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

    def get_project_name(self):
        """
        Method returning the chosen project name, either from QLineEdit (new project) or QComboBox (open project)

        :return: the project name
        :rtype: str
        """
        if self.project_action == ProjectAction.New:
            return self.project_name.text()
        return self.project_name.currentText()

    def accept(self):
        """
        Action triggered by clicking on the accept button in the button box. It is either create a new project
        if the project name does not exist yet or open an existing project otherwise. This method also checks whether
        the project name is empty.
        """
        p_name = self.get_project_name()

        match self.project_action:
            case ProjectAction.Delete:
                msg = f'Deleting {p_name}, please wait.'
                action_method = self.delete_project
            case ProjectAction.New:
                if p_name in project_list():
                    error_msg = QMessageBox(self)
                    error_msg.setText(f'Error: {p_name} project already exists!!!')
                    error_msg.exec()
                    return
                else:
                    msg = f'Creating {p_name}, please wait.'
                    action_method = self.open_create_project
            case ProjectAction.Open:
                msg =f'Opening {p_name}, please wait.'
                action_method = self.open_create_project
            case _:
                return

        print(f'{msg} bis')
        self.wait = WaitDialog(msg, self)
        self.finished.connect(self.wait.close_window)
        self.finished.connect(self.hide)
        self.wait.wait_for(action_method, project_name=p_name)

    def open_create_project(self, project_name):
        """
        Open a project called project_name, create a new project if it does not exist, and set the Window title
        accordingly before closing the project connexion.

        :param project_name: the name of the project to open/create
        :type project_name: str
        """

        try:
            with pydetecdiv_project(project_name) as project:
                PyDetecDiv().project_selected.emit(project.dbname)
                PyDetecDiv().raw_data_counted.emit(project.count_objects('Data'))
            self.finished.emit(True)
        except OpenProjectError as e:
            self.finished.emit(True)
            MessageDialog(e.message)

    def delete_project(self, project_name):
        delete_project(project_name)
        self.finished.emit(True)

class NewProject(QAction):
    """
    Action to open a for project creation
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/new_project"), "&New project", parent)
        # self.triggered.connect(lambda _: ProjectDialog(new_project_dialog=True))
        self.triggered.connect(self.create_project)
        parent.addAction(self)

    def create_project(self):
        try:
            ProjectDialog(project_list(), project_action=ProjectAction.New)
        except UnknownRepositoryTypeError as e:
            MessageDialog(e.message)


class OpenProject(QAction):
    """
    Action to open a project chooser window
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/open_project"), "&Open project", parent)
        self.triggered.connect(self.open_project)
        parent.addAction(self)

    def open_project(self):
        try:
            ProjectDialog(project_list(), project_action=ProjectAction.Open)
        except UnknownRepositoryTypeError as e:
            MessageDialog(e.message)

class DeleteProject(QAction):
    """
    Action to open a project chooser window
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/delete_project"), "&Delete project", parent)
        self.triggered.connect(self.delete_project)
        parent.addAction(self)

    def delete_project(self):
        try:
            ProjectDialog(project_list(), project_action=ProjectAction.Delete)
        except UnknownRepositoryTypeError as e:
            MessageDialog(e.message)
