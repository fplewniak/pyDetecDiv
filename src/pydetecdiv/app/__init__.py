#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Definition of global objects and methods for easy access from all parts of the application
"""
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, QSettings, Slot, QThread

import pydetecdiv.persistence.project
from pydetecdiv.settings import get_config_files
from pydetecdiv.persistence.project import list_projects
from pydetecdiv.domain.Project import Project


class PyDetecDivApplication(QApplication):
    """
    PyDetecDiv application class extending QApplication to keep track of the current project and main window
    """
    project = None
    project_name = None
    main_window = None

    def __init__(self, *args):
        super().__init__(*args)
        self.setApplicationName('pyDetecDiv')

    @classmethod
    def open_project(cls, project_name):
        """
        Open a project from its name if it exists, create it otherwise and store the object in the global cls.project
        class variable
        :param project_name: the project name
        :type project_name: str
        """
        cls.project_name = project_name
        cls.project = Project(project_name)
        cls.main_window.setWindowTitle(f'pyDetecDiv: {project_name}')

    @classmethod
    def close_project(cls):
        cls.project.commit()
        cls.project.repository.close()

class PyDetecDivThread(QThread):
    """
    Thread used to run a process defined by a function and its arguments
    """

    def __init__(self):
        super().__init__()
        self.fn = None
        self.args = None
        self.kwargs = None

    def set_function(self, fn, *args, **kwargs):
        """
        DEfine the function to run in the thread
        :param fn: the function to run
        :param args: arguments passed to the function
        :param kwargs: keyword arguments passed to the function
        """
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        """
        Run the function
        """
        self.fn(*self.args, **self.kwargs)


class WaitDialog(QDialog):
    """
    Generic dialog box asking the user to wait for a thread to be finished. This box closes automatically when the
    thread is complete and the parent window is hidden as well if it is specified. This should be used for processes
    that do not last too long and that might generate inconsistency if cancelled as there is no possibility to interrupt
    it
    """

    def __init__(self, msg='Please wait', thread=None, parent=None):
        super().__init__(parent)
        self.parent = parent
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
        if self.parent is not None:
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
