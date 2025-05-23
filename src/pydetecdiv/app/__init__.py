#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Definition of global objects and methods for easy access from all parts of the application
"""
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pydetecdiv.app.gui.Windows import MainWindow

import os.path
import sys
from collections import defaultdict
from contextlib import contextmanager
from enum import StrEnum
import markdown

from PySide6.QtGui import QCursor, QTextCursor, QCloseEvent
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QVBoxLayout, QProgressBar, QDialogButtonBox, QTextEdit, QWidget)
from PySide6.QtCore import Qt, QSettings, Slot, QThread, Signal, QObject
import pyqtgraph as pg

from pydetecdiv import plugins
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.settings import get_config_file, get_appdata_dir, get_config_value
from pydetecdiv.persistence.project import list_projects
from pydetecdiv.domain.Project import Project


class DrawingTools(StrEnum):
    """
    Enumeration of available drawing tools
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
    saved_rois = Signal(str)
    viewer_roi_click = Signal(tuple)
    scene_modified = Signal(object)
    other_scene_in_focus = Signal(object)
    graphic_item_added = Signal(object)
    graphic_item_deleted = Signal(object)

    version = '0.5.0'
    project_name = None
    main_window = None
    current_drawing_tool = None
    plugin_list = None
    app = None

    roi_template = None
    apply_drift = False

    def __init__(self, *args: list):
        super().__init__(*args)
        self.setApplicationName('pyDetecDiv')
        self.load_plugins()

    @staticmethod
    def load_plugins() -> None:
        """
        Load the available plugins
        """
        PyDetecDiv.plugin_list = plugins.PluginList()
        PyDetecDiv.plugin_list.load()

    @staticmethod
    def set_main_window(main_window: 'MainWindow') -> None:
        """
        Sets the main window global variable to make it accessible across the whole application

        :param main_window: the Main Window object
        """
        PyDetecDiv.main_window = main_window
        PyDetecDiv.main_window.show()

    @staticmethod
    def set_apply_drift(apply_drift: bool) -> None:
        """
        Sets the global switch for drift correction, so it is available over the whole application

        :param apply_drift: the global drift correction switch
        """
        PyDetecDiv.apply_drift = apply_drift


@contextmanager
def pydetecdiv_project(project_name: str) -> Project:
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
        self.func = None
        self.args = None
        self.kwargs = None

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
        self.func(*self.args, **self.kwargs)


class AbstractWaitDialog(QDialog):
    """
    Generic dialog box asking the user to wait for a thread to be finished. This box closes automatically when the
    thread is complete and the parent window is hidden as well if it is specified. This should be used for processes
    that do not last too long and that might generate inconsistency if cancelled as there is no possibility to interrupt
    it
    """

    def __init__(self, parent: QWidget, cancel_msg=None, ignore_close_event=True) -> None:
        super().__init__(parent)
        self.cancel_msg = cancel_msg
        self._ignore_close_event = ignore_close_event
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.pdd_thread = PyDetecDivThread()

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

    def __init__(self, msg, parent: QWidget , progress_bar: bool = False, cancel_msg: str = None, ignore_close_event: bool = True):
        super().__init__(parent, cancel_msg=cancel_msg, ignore_close_event=ignore_close_event)
        self.label = QLabel()
        self.label.setStyleSheet("""
        font-weight: bold;
        """)
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
        self.label.setText(self.cancel_msg)
        super().cancel()


class MessageDialog(QDialog):
    """
    Generic dialog to communicate a message to the user (error, warning or any other information)
    """

    def __init__(self, msg: str):
        super().__init__()
        # self.setWindowModality(Qt.WindowModal)
        label = QLabel()
        label.setStyleSheet("""
        font-weight: bold;
        """)
        label.setText(msg)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)
        self.setLayout(layout)
        self.exec()


class StdoutWaitDialog(AbstractWaitDialog):
    """
    A Wait dialog that also captures and displays stdout output on the fly.
    """
    def __init__(self, msg: str, parent: QWidget, cancel_msg: str = None, ignore_close_event: bool = True):
        super().__init__(parent, cancel_msg=cancel_msg, ignore_close_event=ignore_close_event)
        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.addText(msg)
        layout = QVBoxLayout(self)
        layout.addWidget(self.log)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.button_box.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close_window)
        self.button_box.button(QDialogButtonBox.StandardButton.Close).setEnabled(False)
        if self.cancel_msg:
            self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
            self.button_box.button(QDialogButtonBox.StandardButton.Cancel).clicked(self.cancel)
            self.button_box.button(QDialogButtonBox.StandardButton.Cancel).clicked.connect(self.set_ignore_close_event)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
        self.redirector = StreamRedirector()
        self.redirector.new_text.connect(self.addText)
        sys.stdout = self.redirector

    def addText(self, text: str) -> None:
        """
        Add text to the log window

        :param text: the text to add
        """
        html = markdown.markdown(text, extensions=['tables'])
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertHtml(html)
        self.log.insertHtml('<br>')

    def cancel(self) -> None:
        """
        Set cancelling message and request for interruption of thread so that the running job can cleanly close
        processes and roll back any modification if needed.
        """
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

    def __init__(self):
        super().__init__()

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
        pass  # Required for compatibility with Python's IO system


def get_settings() -> QSettings:
    """
    Get settings in pydetecdiv.ini file

    :return: the settings
    """
    settings = QSettings(str(get_config_file()), QSettings.Format.IniFormat)
    if settings.value("paths/appdata") is None:
        settings.setValue("paths/appdata", get_appdata_dir())
    return settings


# def get_plugins_dir():
#     """
#     Get the user directory where plugins are installed. The directory is created if it does not exist
#     :return: the user plugin path
#     :rtype: Path
#     """
#     plugins_path = os.path.join(get_appdata_dir(), 'plugins')
#     if not os.path.exists(plugins_path):
#         os.mkdir(plugins_path)
#     return plugins_path

def get_project_dir(project_name: str = None) -> str:
    """
    Gets the directory of a project

    :param project_name: the name of the project
    :return: the directory path of the project
    """
    if project_name is None:
        project_name = PyDetecDiv.project_name
    workspace_dir = get_config_value('project', 'workspace')
    return os.path.join(workspace_dir, project_name)


def project_list() -> list[str]:
    """
    Get the list of available projects. This method hides its persistence layer equivalent from other widgets.

    :return: the list of available projects
    """
    return list_projects()


def create_app() -> PyDetecDiv:
    """
    Creates a GUI application (the global controller for the display layer)

    :return: the application
    """
    PyDetecDiv.app = PyDetecDiv([])
    PyDetecDiv.plugin_list.register_all()
    pg.setConfigOptions(antialias=True, background='w')
    return PyDetecDiv.app
