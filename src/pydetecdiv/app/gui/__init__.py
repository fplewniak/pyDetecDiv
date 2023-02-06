#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Main widgets to use with persistent windows
"""
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolBar, QStatusBar, QMenu, QApplication
from  pydetecdiv.app.gui import ActionsSettings, ActionsProject
import pydetecdiv.app.gui.resources


class FileMenu(QMenu):
    """
    The main window File menu
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        menu = parent.menuBar().addMenu("&File")
        ActionsProject.OpenProject(menu).setShortcut("Ctrl+O")
        ActionsProject.NewProject(menu).setShortcut("Ctrl+N")
        menu.addSeparator()
        ActionsSettings.Settings(menu)
        menu.addSeparator()
        Quit(menu).setShortcut("Ctrl+Q")


class MainToolBar(QToolBar):
    """
    The main toolbar of the main window
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName('Main toolbar')
        ActionsProject.OpenProject(self)
        ActionsProject.NewProject(self)
        Quit(self)
        Help(self)


class MainStatusBar(QStatusBar):
    """
    The status bar of the main window
    """
    def __init__(self):
        super().__init__()
        self.setObjectName('Main status bar')


class Quit(QAction):
    """
    Quit action, interrupting the application
    """
    def __init__(self, parent):
        super().__init__(QIcon(":icons/exit"), "&Quit", parent)
        self.triggered.connect(QApplication.quit)
        parent.addAction(self)


class Help(QAction):
    """
    Action requesting global help
    """
    def __init__(self, parent):
        super().__init__(QIcon(":icons/help"), "&Help", parent)
        self.triggered.connect(lambda: print(len(QApplication.allWindows())))
        self.triggered.connect(lambda: print(QApplication.allWindows()))
        parent.addAction(self)
