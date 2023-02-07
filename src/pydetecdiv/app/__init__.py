#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Definition of global objects and methods for easy access from all parts of the application
"""
from contextlib import contextmanager

from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, QSettings, Slot, QThread

from pydetecdiv.settings import get_config_files
from pydetecdiv.persistence.project import list_projects
from pydetecdiv.domain.Project import Project


class PyDetecDivApplication(QApplication):
    """
    PyDetecDiv application class extending QApplication to keep track of the current project and main window
    """
    project_name = None
    main_window = None

    def __init__(self, *args):
        super().__init__(*args)
        self.setApplicationName('pyDetecDiv')

@contextmanager
def pydetecdiv_project(project_name):
    """
    Context manager for projects.
    :param project_name: the project name
    :type project_name: str
    """
    PyDetecDivApplication.project_name = project_name
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
        DEfine the function to run in the thread
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

    def __init__(self, msg='Please wait', thread=None, parent=None, hide_parent=False):
        super().__init__(parent)
        self.parent = parent
        self.hide_parent = hide_parent and parent is not None
        self.setWindowModality(Qt.WindowModal)
        label = QLabel()
        label.setStyleSheet("""
        font-weight: bold;
        """)
        label.setText(msg)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        self.setLayout(layout)
        thread.start()
        thread.finished.connect(self.thread_complete)
        self.exec()
        self.destroy()

    @Slot()
    def thread_complete(self):
        """
        Slot reacting to the end of the thread. The dialog box is destroyed and its parent is hidden if it has been
        specified (not None)
        """
        if self.hide_parent:
            self.parent.hide()
        self.hide()


def get_settings():
    """
    Get settings in pydetecdiv.ini file
    :return: the settings
    :rtype: QSetting instance
    """
    return QSettings(str(get_config_files()[0]), QSettings.IniFormat)


def project_list():
    """
    Get the list of available projects. This method hides its persistence layer equivalent from other widgets.
    :return: the list of available projects
    :rtype: list of str
    """
    return list_projects()
