#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Definition of global objects and methods for easy access from all parts of the application
"""
from typing import TYPE_CHECKING, Callable, Any, Generator

import os.path
import sys
from collections import defaultdict
from contextlib import contextmanager
from enum import StrEnum
import markdown

from PySide6.QtGui import QCursor, QTextCursor, QCloseEvent
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QVBoxLayout, QProgressBar, QDialogButtonBox, QTextEdit, QWidget)
from PySide6.QtCore import Qt, QSettings, Slot, QThread, Signal, QObject, SignalInstance

from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.settings import get_config_file, get_appdata_dir, get_config_value, Device
from pydetecdiv.persistence.project import list_projects
from pydetecdiv.domain.Project import Project

if TYPE_CHECKING:
    from pydetecdiv.app.gui.Windows import MainWindow
    from pydetecdiv.app.gui.SourcePath import TableEditor
    from pydetecdiv.app.tools import Tool


class DrawingTools(StrEnum):
    """
    Enumeration of available drawing new_tools
    """
    Cursor = 'Select/move'
    DrawRect = 'Draw Rectangle'
    DuplicateItem = 'Duplicate selected Item'
    DrawPoint = 'Draw point'


class PyDetecDiv(QApplication):
    """
    PyDetecDiv application class extending QApplication to keep track of the current project and main window
    """
    project_selected = Signal(str)
    raw_data_counted = Signal(int)
    roi_counted = Signal(int)
    saved_rois = Signal(str)
    viewer_roi_click = Signal(tuple)
    scene_modified = Signal(object)
    other_scene_in_focus = Signal(object)
    graphic_item_added = Signal(object)
    graphic_item_deleted = Signal(object)

    version = '0.7.0'
    project_name = None
    main_window = None
    current_drawing_tool = None
    # plugin_list = None
    app = None
    tools = {}

    roi_template = None
    apply_drift = False

    def __init__(self, *args: list):
        super().__init__(*args)
        self.setApplicationName('pyDetecDiv')

    @staticmethod
    def update_tools(new_tools: dict[str, 'Tool']) -> None:
        """
        Update the list of tools installed in the application
        :param new_tools: the dictionary declaring tools
        """
        PyDetecDiv.tools.update(new_tools)

    @staticmethod
    def check_data_source_paths(table_editor: 'TableEditor') -> None:
        """
        Checks the data source path configuration and proposes a dialog window to edit any definition that may be required for
        use of shared data sources on the current device

        :param table_editor:
        """
        df = Device.undefined_paths()

        if not df.is_empty():
            for grp in df.group_by(by='path_id'):
                table_editor.set_data(grp[1].select(['name', 'device', 'path', 'path_id'])).hide_columns(['path_id'])
                table_editor.exec()

    @staticmethod
    def set_main_window(main_window: 'MainWindow') -> 'MainWindow':
        """
        Sets the main window global variable to make it accessible across the whole application

        :param main_window: the Main Window object
        """
        PyDetecDiv.main_window = main_window
        PyDetecDiv.main_window.show()
        return PyDetecDiv.main_window

    @staticmethod
    def set_apply_drift(apply_drift: bool) -> None:
        """
        Sets the global switch for drift correction, so it is available over the whole application

        :param apply_drift: the global drift correction switch
        """
        PyDetecDiv.apply_drift = apply_drift


@contextmanager
def pydetecdiv_project(project_name: str) -> Generator[Project, Any, None]:
    """
    Context manager for projects.

    :param project_name: the project name
    """
    PyDetecDiv.project_name = project_name
    project = Project(project_name)
    try:
        yield project
    finally:
        project.commit()
        project.repository.close()
        project.pool = defaultdict(DomainSpecificObject)


class MessageDialog(QDialog):
    """
    Generic dialog to communicate a message to the user (error, warning or any other information)
    """

    def __init__(self, msg: str, html: bool = True):
        super().__init__()
        # self.setWindowModality(Qt.WindowModal)
        label = QLabel()
        label.setText(msg)
        if html:
            label.setTextFormat(Qt.TextFormat.RichText)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)
        self.setLayout(layout)
        self.exec()


class ConfirmDialog(QDialog):
    """
    Generic dialog asking for confirmation from the user to launch an action
    """

    def __init__(self, msg: str, action: Callable):
        super().__init__()
        # self.setWindowModality(Qt.WindowModal)
        self.action = action
        label = QLabel()
        # label.setStyleSheet("""
        # font-weight: bold;
        # """)
        label.setText(msg)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)
        self.setLayout(layout)
        self.exec()

    def accept(self, /):
        """
        Close the window and launch action
        """
        self.close()
        self.action()


def get_settings() -> QSettings:
    """
    Get settings in pydetecdiv.ini file

    :return: the settings
    """
    settings = QSettings(str(get_config_file()), QSettings.Format.IniFormat)
    if settings.value("paths/appdata") is None:
        settings.setValue("paths/appdata", get_appdata_dir())
    return settings


def get_project_dir(project_name: str | None = None) -> str:
    """
    Gets the directory of a project

    :param project_name: the name of the project
    :return: the directory path of the project
    """
    if project_name is None:
        project_name = PyDetecDiv.project_name
    workspace_dir = get_config_value('project', 'workspace')
    if project_name is None:
        return workspace_dir
    return os.path.join(workspace_dir, project_name)


def project_list() -> list[str]:
    """
    Get the list of available projects. This method hides its persistence layer equivalent from other widgets.

    :return: the list of available projects
    """
    return list_projects()

def set_connections(connections: dict[SignalInstance, Callable | list[Callable]]) -> None:
    """
    connect a signal to a slot or a list of slots, as defined in a dictionary

    :param connections: the dictionary {signal: slot,...} or {signal: [slot1, slot2,...],...} containing the connections
     to create
    """
    for signal, slot in connections.items():
        if isinstance(slot, list):
            for s in slot:
                signal.connect(s)
        else:
            signal.connect(slot)
