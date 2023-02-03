#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Definition of global objects and methods for easy access from all parts of the application
"""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from pydetecdiv.settings import get_config_files
from pydetecdiv.persistence.project import list_projects
from pydetecdiv.domain.Project import Project


class PyDetecDivApplication(QApplication):
    """
    PyDetecDiv application class extending QApplication to keep track of the current project and main window
    """
    project = None
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
        cls.project = Project(project_name)


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
