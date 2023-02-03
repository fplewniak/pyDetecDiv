"""
Handling actions to open, create and interact with projects
"""
from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QLineEdit, QDialogButtonBox, QComboBox, QMessageBox, QDialog,
                               QInputDialog)
from pydetecdiv.app import PyDetecDivApplication, project_list


class NewProjectDialog(QDialog):
    """
    A dialog window to create a new project
    """

    def __init__(self):
        super().__init__(PyDetecDivApplication.main_window)

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

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.hide)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.project_name)
        self.layout.addWidget(self.button_box)

        self.exec()
        self.destroy()

    def accept(self):
        """
        When the OK button has been clicked, checks the project name is not empty and that the project does not exist
        before creating the project
        """
        p_name = self.project_name.text()
        if p_name in project_list():
            error_msg = QMessageBox(self)
            error_msg.setText(f'Error: {p_name} project already exists!!!')
            error_msg.exec()
        elif len(p_name) == 0:
            ...
        else:
            self.label.setText(f'Creating {p_name}, please wait.')
            self.project_name.setDisabled(True)
            self.button_box.setDisabled(True)
            self.adjustSize()
            self.repaint()
            PyDetecDivApplication.open_project(p_name)
            PyDetecDivApplication.main_window.setWindowTitle(f'pyDetecDiv: {p_name}')

            self.hide()


class NewProject(QAction):
    """
    Action to open a for project creation
    """

    def __init__(self, parent, icon=False):
        super().__init__("&New project", parent)
        if icon:
            self.setIcon(QIcon(":icons/new_project"))
        self.triggered.connect(NewProjectDialog)
        parent.addAction(self)

    def create_project(self):
        """
        A method to create a project using a QInputDialog instance instead of the NewProjectDialog class above. This
        method is not used at the moment but might be in the near future
        """
        dialog = QInputDialog()
        dialog.setWindowTitle('Create project')
        dialog.setLabelText('Enter a name for your new project:')
        dialog.setTextValue('MyProject')
        ok = dialog.exec()
        if ok:
            p_name = dialog.textValue()
            if p_name in project_list():
                error_msg = QMessageBox(self)
                error_msg.setText(f'Error: {p_name} project already exists!!!')
                error_msg.exec()
            elif len(p_name) == 0:
                ...
            else:
                self.setDisabled(True)
                self.repaint()
                PyDetecDivApplication.open_project(p_name)
                PyDetecDivApplication.main_window.setWindowTitle(f'pyDetecDiv: {p_name}')


class OpenProjectDialog(QDialog):
    """
    A dialog window to open a project
    """

    def __init__(self):
        super().__init__(PyDetecDivApplication.main_window)
        self.setWindowModality(Qt.WindowModal)
        self.setWindowTitle('Open project')

        self.layout = QVBoxLayout(self)
        self.label = QLabel('Select a project name:')
        self.combo = QComboBox()
        self.combo.addItems(sorted(project_list()))

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.hide)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.combo)
        self.layout.addWidget(self.button_box)

        self.exec()
        self.destroy(True)

    def accept(self):
        """
        Open the project when the OK button has been clicked
        """
        PyDetecDivApplication.open_project(self.combo.currentText())
        PyDetecDivApplication.main_window.setWindowTitle(f'pyDetecDiv: {PyDetecDivApplication.project.dbname}')
        self.hide()


class OpenProject(QAction):
    """
    Action to open a project chooser window
    """

    def __init__(self, parent, icon=False):
        super().__init__("&Open project", parent)
        if icon:
            self.setIcon(QIcon(":icons/open_project"))
            # self.setIcon(QIcon.fromTheme('folder-open'))
        self.triggered.connect(OpenProjectDialog)
        # self.triggered.connect(self.open_project)
        parent.addAction(self)

    def open_project(self):
        """
        A method to open a project using a QInputDialog instance instead of the OpenProjectDialog class above. This
        method is not used at the moment but might be in the near future.
        """
        dialog = QInputDialog()
        dialog.setWindowTitle('Open project')
        dialog.setLabelText('Select a project name:')
        dialog.setComboBoxItems(sorted(project_list()))
        dialog.setComboBoxEditable(False)
        ok = dialog.exec()
        if ok:
            PyDetecDivApplication.open_project(dialog.textValue())
            PyDetecDivApplication.main_window.setWindowTitle(f'pyDetecDiv: {PyDetecDivApplication.project.dbname}')
