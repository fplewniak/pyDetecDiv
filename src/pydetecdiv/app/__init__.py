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


class PyDetecDivThread(QThread):
    """
    Thread used to run a process defined by a function and its arguments
    """

    def __init__(self):
        super().__init__()
        self.func: Callable | None = None
        self.args: list = []
        self.kwargs: dict = {}

    def set_function(self, func: Callable, *args: list, **kwargs: dict) -> None:
        """
        Define the function to run in the thread

        :param func: the function to run
        :param args: arguments passed to the function
        :param kwargs: keyword arguments passed to the function
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self) -> None:
        """
        Run the function
        """
        if self.func is not None:
            self.func(*self.args, **self.kwargs)


class AbstractWaitDialog(QDialog):
    """
    Generic dialog box asking the user to wait for a thread to be finished. This box closes automatically when the
    thread is complete and the parent window is hidden as well if it is specified. This should be used for processes
    that do not last too long and that might generate inconsistency if cancelled as there is no possibility to interrupt
    it
    """

    def __init__(self, parent: QWidget, title: str | None = None, cancel_msg: str | None = None,
                 ignore_close_event: bool= True, close_when_finished: bool = True) -> None:
        super().__init__(parent)
        if title is not None:
            self.setWindowTitle(title)
        else:
            self.setWindowTitle(PyDetecDiv.project_name)
        self.cancel_msg = cancel_msg
        self._ignore_close_event = ignore_close_event
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.pdd_thread = PyDetecDivThread()
        self.parent = parent
        if hasattr(self.parent, 'finished') and close_when_finished:
            self.parent.finished.connect(self.close_window)

    def wait_for(self, func: Callable, *args: list, **kwargs: dict) -> None:
        """
        Run function in separate thread and launch local event loop to handle progress bar and cancellation

        :param func: the function to run
        :param args: positional arguments for the function
        :param kwargs: keyword arguments for the function
        """
        PyDetecDiv.app.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        self.pdd_thread.set_function(func, *args, **kwargs)
        self.pdd_thread.start()
        self.exec()

    def close_window(self) -> None:
        """
        Hide and destroy the Wait dialog window. The cursor is also set back to its normal aspect.
        """
        self.hide()
        PyDetecDiv.app.restoreOverrideCursor()
        self.destroy()

    def cancel(self) -> None:
        """
        Set cancelling message and request for interruption of thread so that the running job can cleanly close
        processes and roll back any modification if needed.
        """
        if self.pdd_thread.isRunning():
            self.pdd_thread.requestInterruption()

    def set_ignore_close_event(self, ignore_close_event: bool = True) -> None:
        """
        Set the _ignore_close_event flag to prevent or allow closing the window

        :param ignore_close_event: value to set the flag to
        """
        self._ignore_close_event = ignore_close_event

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Cancel the job if the window is closed unless close event is ignored by request.

        :param event: close event
        """
        if self._ignore_close_event:
            event.ignore()
        else:
            self.cancel()


class WaitDialog(AbstractWaitDialog):
    """
    Generic dialog box asking the user to wait for a thread to be finished. This box closes automatically when the
    thread is complete and the parent window is hidden as well if it is specified. This should be used for processes
    that do not last too long and that might generate inconsistency if cancelled as there is no possibility to interrupt
    it
    """

    def __init__(self, msg, parent: QWidget, title: str | None = None, progress_bar: bool = False, cancel_msg: str | None= None,
                 ignore_close_event: bool = True, close_when_finished: bool = True):
        super().__init__(parent, title=title, cancel_msg=cancel_msg, ignore_close_event=ignore_close_event,
                         close_when_finished=close_when_finished)
        if hasattr(self.parent, 'progress'):
            self.parent.progress.connect(self.show_progress)

        self.label = QLabel()
        # self.label.setStyleSheet("""
        # font-weight: bold;
        # """)
        self.label.setText(msg)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        if progress_bar:
            self.progress_bar_widget = QProgressBar()
            layout.addWidget(self.progress_bar_widget)
        if cancel_msg:
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
            button_box.rejected.connect(self.cancel)
            button_box.rejected.connect(button_box.hide)
            button_box.rejected.connect(self.set_ignore_close_event)
            layout.addWidget(button_box)
        self.setLayout(layout)

    def show_progress(self, i: int) -> None:
        """
        Convenience method to send the progress value to the progress bar widget

        :param i: the value to pass to the progress bar
        """
        self.progress_bar_widget.setValue(i)

    def cancel(self) -> None:
        """
        Set cancelling message and request for interruption of thread so that the running job can cleanly close
        processes and roll back any modification if needed.
        """
        if self.cancel_msg:
            self.label.setText(self.cancel_msg)
        super().cancel()


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


class StdoutWaitDialog(AbstractWaitDialog):
    """
    A Wait dialog that also captures and displays stdout output on the fly.
    """

    def __init__(self, msg: str, parent: QWidget, cancel_msg: str | None = None, ignore_close_event: bool = True,
                 close_when_finished: bool = True):
        super().__init__(parent, cancel_msg=cancel_msg, ignore_close_event=ignore_close_event,
                         close_when_finished=close_when_finished)
        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setHtml(markdown.markdown(msg))
        layout = QVBoxLayout(self)
        layout.addWidget(self.log)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.button_box.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close_window)
        self.button_box.button(QDialogButtonBox.StandardButton.Close).setEnabled(False)
        if self.cancel_msg:
            self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
            self.button_box.button(QDialogButtonBox.StandardButton.Cancel).clicked.connect(self.cancel)
            self.button_box.button(QDialogButtonBox.StandardButton.Cancel).clicked.connect(self.set_ignore_close_event)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
        self.redirector = StreamRedirector()
        self.redirector.new_text.connect(self.addHtmlText)
        sys.stdout = self.redirector

    def addText(self, text: str) -> None:
        """
        Add text to the log window

        :param text: the text to add
        """
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        if hasattr(self.parent, 'tool'):
            self.parent.tool.log_text(text)

    def addHtmlText(self, text: str) -> None:
        """
        Add text to the log window

        :param text: the text to add
        """
        html = markdown.markdown(text, extensions=['tables'])
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertHtml(html)
        self.log.insertHtml('<br>')
        if hasattr(self.parent, 'tool'):
            self.parent.tool.log_text(text)

    def cancel(self) -> None:
        """
        Set cancelling message and request for interruption of thread so that the running job can cleanly close
        processes and roll back any modification if needed.
        """
        if self.cancel_msg:
            self.log.append(self.cancel_msg)
        super().cancel()

    def close_window(self) -> None:
        """
        Closes the window and stops stdout capture
        """
        sys.stdout = sys.__stdout__
        super().close_window()

    def stop_redirection(self, signal: Signal) -> None:
        """
        Stops capturing the stdout output, which is therefore printed to the terminal again

        :param signal: the signal triggered by the event requesting to stop redirection
        """
        if self.cancel_msg:
            self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setEnabled(False)
        self.button_box.button(QDialogButtonBox.StandardButton.Close).setEnabled(True)
        PyDetecDiv.app.restoreOverrideCursor()
        sys.stdout = sys.__stdout__


class StreamRedirector(QObject):
    """Custom stream redirector to emit stdout/stderr output."""
    new_text = Signal(str)

    def write(self, text: str) -> None:
        """
        Write text to the stream redirector

        :param text: text to be written
        """
        self.new_text.emit(text)

    def flush(self) -> None:
        """
        A dummy method required only for compatibility with the Python IO system
        """
        # Required for compatibility with Python's IO system


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
