"""
Handling actions to open, create and interact with projects
"""
from PySide6.QtCore import Qt, QRegularExpression, Slot, Signal
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QLineEdit, QDialogButtonBox, QComboBox, QMessageBox, QDialog,)
from pydetecdiv.app import PyDetecDiv, project_list, WaitDialog, pydetecdiv_project

class ProjectDialog(QDialog):
    """
    A generic dialog window to create or open a project
    """
    finished = Signal(bool, name='projectOpen')

    def __init__(self, new_project_dialog=False):
        super().__init__(PyDetecDiv().main_window)
        self.wait = None
        self.new_project_dialog = new_project_dialog
        self.setWindowModality(Qt.WindowModal)
        self.layout = QVBoxLayout(self)

        if self.new_project_dialog:
            self.setWindowTitle('Create project')
            self.label = QLabel('Enter a name for your new project:')
            self.project_name = QLineEdit('MyProject')
            self.project_name.textChanged.connect(self.project_name_changed)
        else:
            self.setWindowTitle('Open project')
            self.label = QLabel('Select a project name:')
            self.project_name = QComboBox()
            self.project_name.addItems(sorted(project_list()))
            self.project_name.setEditable(True)
            self.project_name.editTextChanged.connect(self.project_name_changed)

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
        if self.new_project_dialog:
            return self.project_name.text()
        return self.project_name.currentText()

    def accept(self):
        """
        Action triggered by clicking on the accept button in the button box. It is either create a new project
        if the project name does not exist yet or open an existing project otherwise. This method also checks whether
        the project name is empty.
        """
        p_name = self.get_project_name()
        if self.new_project_dialog:
            if p_name in project_list():
                error_msg = QMessageBox(self)
                error_msg.setText(f'Error: {p_name} project already exists!!!')
                error_msg.exec()
            else:
                self.wait = WaitDialog(f'Creating {p_name}, please wait.', self)
        else:
            self.wait = WaitDialog(f'Opening {p_name}, please wait.', self)
        self.finished.connect(self.wait.close_window)
        self.finished.connect(self.hide)
        self.wait.wait_for(self.open_create_project, project_name=p_name)
        PyDetecDiv().project_selected.emit(p_name)

    def open_create_project(self, project_name):
        """
        Open a project called project_name, create a new project if it does not exist, and set the Window title
        accordingly before closing the project connexion.
        :param project_name: the name of the project to open/create
        :type project_name: str
        """
        with pydetecdiv_project(project_name) as project:
            PyDetecDiv().main_window.setWindowTitle(f'pyDetecDiv: {project.dbname}')
        self.finished.emit(True)


class NewProject(QAction):
    """
    Action to open a for project creation
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/new_project"), "&New project", parent)
        self.triggered.connect(lambda _: ProjectDialog(new_project_dialog=True))
        parent.addAction(self)


class OpenProject(QAction):
    """
    Action to open a project chooser window
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/open_project"), "&Open project", parent)
        self.triggered.connect(ProjectDialog)
        parent.addAction(self)
