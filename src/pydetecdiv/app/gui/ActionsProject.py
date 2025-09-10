"""
Handling actions to open, create and interact with projects
"""
import os
from enum import Enum
import polars

from PySide6.QtCore import Qt, QRegularExpression, Slot, Signal
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QLineEdit, QDialogButtonBox, QComboBox, QMessageBox, QDialog, QWidget, )
from pydetecdiv.app import PyDetecDiv, project_list, WaitDialog, pydetecdiv_project, ConfirmDialog
from pydetecdiv.app import MessageDialog
from pydetecdiv.app.gui.SourcePath import TableEditor
from pydetecdiv.persistence.project import delete_project
from pydetecdiv.exceptions import OpenProjectError, UnknownRepositoryTypeError
from pydetecdiv.settings import Device


class ProjectAction(Enum):
    """
    An Enum class of the various actions applicable to projects (New, Open, Delete)
    """
    New = 1
    Open = 2
    Delete = 3


class ProjectDialog(QDialog):
    """
    A generic dialog window to create or open a project
    """
    finished = Signal(bool, name='projectOpen')

    def __init__(self, list_projects: list[str], project_action: ProjectAction = ProjectAction.Open):
        super().__init__(PyDetecDiv.main_window)
        self.wait = None
        self.project_action = project_action
        self.setWindowModality(Qt.WindowModality.WindowModal)
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
                self.project_name.addItems(sorted(list_projects))
                match self.project_action:
                    case ProjectAction.Open:
                        self.setWindowTitle('Open project')
                        self.project_name.setEditable(True)
                        self.project_name.editTextChanged.connect(self.project_name_changed)
                    case ProjectAction.Delete:
                        self.setWindowTitle('Delete project')

        self.project_name.setValidator(self.project_name_validator())

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                                           Qt.Orientation.Horizontal)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.hide)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.project_name)
        self.layout.addWidget(self.button_box)

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy()

    @staticmethod
    def project_name_validator() -> QRegularExpressionValidator:
        """
        Name validator to filter invalid character in project name

        :return: the validator
        """
        name_filter = QRegularExpression()
        name_filter.setPattern('\\w[\\w-]*')
        validator = QRegularExpressionValidator()
        validator.setRegularExpression(name_filter)
        return validator

    @Slot()
    def project_name_changed(self) -> None:
        """
        Slot checking whether the project name input is empty or not and enabling or disabling Ok button accordingly
        """
        if self.get_project_name():
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def get_project_name(self) -> str:
        """
        Method returning the chosen project name, either from QLineEdit (new project) or QComboBox (open project)

        :return: the project name
        """
        if isinstance(self.project_name, QLineEdit):
            return self.project_name.text()
        return self.project_name.currentText()

    def accept(self) -> None:
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
                msg = f'Opening {p_name}, please wait.'
                action_method = self.open_create_project
            case _:
                return

        self.wait = WaitDialog(msg, self)
        self.finished.connect(self.wait.close_window)
        self.finished.connect(self.hide)
        self.wait.wait_for(action_method, project_name=p_name)

    def open_create_project(self, project_name: str) -> None:
        """
        Open a project called project_name, create a new project if it does not exist, and set the Window title
        accordingly before closing the project connexion.

        :param project_name: the name of the project to open/create
        """

        try:
            with pydetecdiv_project(project_name) as project:
                PyDetecDiv.app.project_selected.emit(project.dbname)
                PyDetecDiv.app.raw_data_counted.emit(project.count_objects('Data'))
            self.finished.emit(True)
        except OpenProjectError as e:
            self.finished.emit(True)
            MessageDialog(e.message)

    def delete_project(self, project_name: str) -> None:
        """
        Delete project called project_name,

        :param project_name: the name of the project to delete
        """
        delete_project(project_name)
        self.finished.emit(True)


class NewProject(QAction):
    """
    Action to open a for project creation
    """

    def __init__(self, parent: QWidget):
        super().__init__(QIcon(":icons/new_project"), "&New project", parent)
        # self.triggered.connect(lambda _: ProjectDialog(new_project_dialog=True))
        self.triggered.connect(self.create_project)
        parent.addAction(self)

    @staticmethod
    def create_project() -> None:
        """
        Opens dialog window to create a new project
        """
        try:
            ProjectDialog(project_list(), project_action=ProjectAction.New)
        except UnknownRepositoryTypeError as e:
            MessageDialog(e.message)


class OpenProject(QAction):
    """
    Action to open a project chooser window
    """

    def __init__(self, parent: QWidget):
        super().__init__(QIcon(":icons/open_project"), "&Open project", parent)
        self.triggered.connect(self.open_project)
        parent.addAction(self)

    def open_project(self) -> None:
        """
        Opens dialog window to open a project
        """
        try:
            ProjectDialog(project_list(), project_action=ProjectAction.Open)
        except UnknownRepositoryTypeError as e:
            MessageDialog(e.message)


class DeleteProject(QAction):
    """
    Action to delete a project chooser window
    """

    def __init__(self, parent: QWidget):
        super().__init__(QIcon(":icons/delete_project"), "&Delete project", parent)
        self.triggered.connect(self.delete_project)
        parent.addAction(self)

    def delete_project(self) -> None:
        try:
            ProjectDialog(project_list(), project_action=ProjectAction.Delete)
        except UnknownRepositoryTypeError as e:
            MessageDialog(e.message)


class ConvertProjectSourceDir(QAction):
    """
    Action to convert the local data source directory to shared source
    """

    def __init__(self, parent: QWidget):
        super().__init__(QIcon(":icons/delete_project"), "Convert to shared data source", parent)
        self.triggered.connect(self.confirm_conversion)
        self.setEnabled(False)
        parent.addAction(self)

    def confirm_conversion(self) -> None:
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            ConfirmDialog(f'You are about to convert {project.dbname} data source path to shared', self.convert_to_shared)

    def convert_to_shared(self) -> None:
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            data_list = polars.from_dicts(project.get_records('Data'))

            print('Checking shared data source paths are defined on this device')
            undefined_paths = Device.undefined_paths().join(data_list, left_on='path_id', right_on='source_dir', how='semi')
            table_editor = TableEditor('Undefined source path',
                                       description=f'This source is used in {project.dbname} but is not configured on this device. '
                                                   'Please configure it to avoid inconsistency.'
                                                   '\nSource path definition on other devices:', force_resolution=True)
            for grp in undefined_paths.group_by(by='path_id'):
                table_editor.set_data(grp[1])
                table_editor.exec()

            print('Checking the data urls are all valid on this device')
            wrong_paths = polars.DataFrame({'path': []}, schema={'path': str})
            for data_object in project.get_objects("Data"):
                if not os.path.isfile(data_object.url):
                    head, tail = os.path.split(data_object.url)
                    while not os.path.isdir(head) and head != '/':
                        head, tail = os.path.split(head)
                    wrong_paths = wrong_paths.extend(polars.DataFrame({'path': [os.path.join(head, tail)]})).unique()
            for path in wrong_paths.rows():
                MessageDialog(f'The path\n{path[0]}\n does not exist on this device\n'
                              f'You should fix that before continuing as this may cause severe inconsistencies')

            print('Searching for a valid shared path for this device')
            source_dir_list = [s for s in data_list['source_dir'].unique() if os.path.isdir(s) and Device.path_id(s) is not None]

            for id_, url, source_dir in data_list.select(['id_', 'url', 'source_dir']
                                                         ).filter(polars.col('source_dir').is_in(source_dir_list)).rows():
                data_object = project.get_object('Data', id_=id_)
                data_object.url_ = os.path.relpath(url, start=Device.data_path(Device.path_id(source_dir)))
                data_object.source_dir = Device.path_id(source_dir)
                data_object.validate(updated=True)

            print('Check there are no local paths left')
