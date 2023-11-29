#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Definition of global objects and methods for easy access from all parts of the application
"""
import os.path
from contextlib import contextmanager
from enum import StrEnum

from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout, QProgressBar, QDialogButtonBox
from PySide6.QtCore import Qt, QSettings, Slot, QThread, Signal

from pydetecdiv import plugins
from pydetecdiv.settings import get_config_file, get_appdata_dir
from pydetecdiv.persistence.project import list_projects
from pydetecdiv.domain.Project import Project
from pydetecdiv.utils import singleton


class DrawingTools(StrEnum):
    """
    Enumeration of available drawing tools
    """
    Cursor = 'Select/move'
    DrawROI = 'Draw ROI'
    DuplicateROI = 'Duplicate selected ROI'


@singleton
class PyDetecDiv(QApplication):
    """
    PyDetecDiv application class extending QApplication to keep track of the current project and main window
    """
    project_selected = Signal(str)
    raw_data_counted = Signal(int)
    saved_rois = Signal(str)

    def __init__(self, *args):
        super().__init__(*args)
        self.setApplicationName('pyDetecDiv')
        self.project_name = None
        self.main_window = None
        self.current_drawing_tool = None
        self.load_plugins()

    def load_plugins(self):
        """
        Load the available plugins
        """
        self.plugin_list = plugins.PluginList()
        self.plugin_list.load()


@contextmanager
def pydetecdiv_project(project_name):
    """
    Context manager for projects.

    :param project_name: the project name
    :type project_name: str
    """
    PyDetecDiv().project_name = project_name
    project = Project(project_name)
    try:
        yield project
    finally:
        project.commit()
        project.repository.close()


class PyDetecDivThread(QThread):
    """
    Thread used to run a process defined by a function and its arguments
    """

    def __init__(self):
        super().__init__()
        self.func = None
        self.args = None
        self.kwargs = None

    def set_function(self, func, *args, **kwargs):
        """
        Define the function to run in the thread

        :param fn: the function to run
        :param args: arguments passed to the function
        :param kwargs: keyword arguments passed to the function
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        """
        Run the function
        """
        self.func(*self.args, **self.kwargs)


class WaitDialog(QDialog):
    """
    Generic dialog box asking the user to wait for a thread to be finished. This box closes automatically when the
    thread is complete and the parent window is hidden as well if it is specified. This should be used for processes
    that do not last too long and that might generate inconsistency if cancelled as there is no possibility to interrupt
    it
    """

    def __init__(self, msg, parent, progress_bar=False, cancel_msg=None, ignore_close_event=True):
        super().__init__(parent)
        self.cancel_msg = cancel_msg
        self._ignore_close_event = ignore_close_event
        self.setWindowModality(Qt.WindowModal)
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
            button_box = QDialogButtonBox(QDialogButtonBox.Cancel, self)
            button_box.rejected.connect(self.cancel)
            button_box.rejected.connect(button_box.hide)
            button_box.rejected.connect(self.set_ignore_close_event)
            layout.addWidget(button_box)
        self.setLayout(layout)
        self.pdd_thread = PyDetecDivThread()

    def show_progress(self, i):
        """
        Convenience method to send the progress value to the progress bar widget

        :param i: the value to pass to the progress bar
        :type i: int
        """
        self.progress_bar_widget.setValue(i)

    def wait_for(self, func, *args, **kwargs):
        """
        Run function in separate thread and launch local event loop to handle progress bar and cancellation

        :param func: the function to run
        :param args: positional arguments for the function
        :param kwargs: keyword arguments for the function
        """
        PyDetecDiv().setOverrideCursor(QCursor(Qt.WaitCursor))
        self.pdd_thread.set_function(func, *args, **kwargs)
        self.pdd_thread.start()
        self.exec()

    def close_window(self):
        """
        Hide and destroy the Wait dialog window. The cursor is also set back to its normal aspect.
        """
        self.hide()
        PyDetecDiv().restoreOverrideCursor()
        self.destroy()

    def cancel(self):
        """
        Set cancelling message and request for interruption of thread so that the running job can cleanly close
        processes and roll back any modification if needed.
        """
        self.label.setText(self.cancel_msg)
        if self.pdd_thread.isRunning():
            self.pdd_thread.requestInterruption()

    def set_ignore_close_event(self, ignore_close_event=True):
        """
        Set the _ignore_close_event flag to prevent or allow closing the window

        :param ignore_close_event: value to set the flag to
        :type ignore_close_event: bool
        """
        self._ignore_close_event = ignore_close_event

    def closeEvent(self, event):
        """
        Cancel the job if the window is closed unless close event is ignored by request.

        :param event: close event
        """
        if self._ignore_close_event:
            event.ignore()
        else:
            self.cancel()


class MessageDialog(QDialog):
    """
    Generic dialog to communicate a message to the user (error, warning or any other information)
    """

    def __init__(self, msg):
        super().__init__()
        self.setWindowModality(Qt.WindowModal)
        label = QLabel()
        label.setStyleSheet("""
        font-weight: bold;
        """)
        label.setText(msg)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        button_box = QDialogButtonBox(QDialogButtonBox.Close, self)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)
        self.setLayout(layout)
        self.exec()


def get_settings():
    """
    Get settings in pydetecdiv.ini file

    :return: the settings
    :rtype: QSetting instance
    """
    settings = QSettings(str(get_config_file()), QSettings.IniFormat)
    if settings.value("paths/appdata") is None:
        settings.setValue("paths/appdata", get_appdata_dir())
    return settings


def get_plugins_dir():
    """
    Get the user directory where plugins are installed. The directory is created if it does not exist
    :return: the user plugin path
    :rtype: Path
    """
    plugins_path = os.path.join(get_appdata_dir(), 'plugins')
    if not os.path.exists(plugins_path):
        os.mkdir(plugins_path)
    return plugins_path


def project_list():
    """
    Get the list of available projects. This method hides its persistence layer equivalent from other widgets.

    :return: the list of available projects
    :rtype: list of str
    """
    return list_projects()
