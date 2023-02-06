"""
Handling actions to open, create and interact with projects
"""
from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QLineEdit, QDialogButtonBox, QComboBox, QMessageBox, QDialog)
from pydetecdiv.app import PyDetecDivApplication, project_list


class ProjectDialog(QDialog):
    """
    A generic dialog window to create or open a project
    """

    def __init__(self, new_project_dialog=False):
        super().__init__(PyDetecDivApplication.main_window)
        self.new_project_dialog = new_project_dialog
        self.setWindowModality(Qt.WindowModal)
        self.layout = QVBoxLayout(self)

        if self.new_project_dialog:
            self.setWindowTitle('Create project')
            self.label = QLabel('Enter a name for your new project:')
            self.project_name = QLineEdit('MyProject')
        else:
            self.setWindowTitle('Open project')
            self.label = QLabel('Select a project name:')
            self.project_name = QComboBox()
            self.project_name.addItems(sorted(project_list()))
            self.project_name.setEditable(True)

        self.project_name.setValidator(self.project_name_validator())

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.hide)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.project_name)
        self.layout.addWidget(self.button_box)

        self.exec()
        self.destroy()

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
        if len(p_name) == 0:
            ...
        elif p_name not in project_list():
            self.label.setStyleSheet("""
            font-weight: bold;
            """)
            self.label.setText(f'Creating {p_name}, please wait.')
            self.project_name.setDisabled(True)
            self.button_box.setDisabled(True)
            self.adjustSize()
            self.repaint()
            PyDetecDivApplication.open_project(p_name)
            PyDetecDivApplication.main_window.setWindowTitle(f'pyDetecDiv: {p_name}')
        elif self.new_project_dialog:
            error_msg = QMessageBox(self)
            error_msg.setText(f'Error: {p_name} project already exists!!!')
            error_msg.exec()
        else:
            PyDetecDivApplication.open_project(p_name)
            PyDetecDivApplication.main_window.setWindowTitle(f'pyDetecDiv: {p_name}')

        self.hide()

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


class NewProject(QAction):
    """
    Action to open a for project creation
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/new_project"), "&New project", parent)
        self.triggered.connect(self.new_project)
        parent.addAction(self)

    @staticmethod
    def new_project():
        """
        Open a project creation dialog window
        """
        ProjectDialog(new_project_dialog=True)


class OpenProject(QAction):
    """
    Action to open a project chooser window
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/open_project"), "&Open project", parent)
        self.triggered.connect(self.open_project)
        parent.addAction(self)

    @staticmethod
    def open_project():
        """
        Open an open/create project dialog
        """
        ProjectDialog()
