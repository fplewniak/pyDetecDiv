#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolBar, QStatusBar, QMenu, QApplication
import pydetecdiv.app.gui.ActionsSettings
import pydetecdiv.app.gui.ActionsProject
import pydetecdiv.app.gui.resources


class FileMenu(QMenu):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        menu = parent.menuBar().addMenu("&File")
        ActionsProject.OpenProject(menu, icon=True).setShortcut("Ctrl+O")
        ActionsProject.NewProject(menu, icon=True).setShortcut("Ctrl+N")
        menu.addSeparator()
        ActionsSettings.Settings(menu)
        menu.addSeparator()
        Quit(menu).setShortcut("Ctrl+Q")


class MainToolBar(QToolBar):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName('Main toolbar')
        ActionsProject.OpenProject(self, icon=True)
        ActionsProject.NewProject(self, icon=True)
        Quit(self, icon=True)
        Help(self, icon=True)


class MainStatusBar(QStatusBar):
    def __init__(self):
        super().__init__()
        self.setObjectName('Main status bar')


class Quit(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&Quit", parent)
        if icon:
            self.setIcon(QIcon(":icons/exit"))
            # self.setIcon(QIcon.fromTheme('application-exit'))
        self.triggered.connect(QApplication.quit)
        parent.addAction(self)


class Help(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&Help", parent)
        if icon:
            self.setIcon(QIcon(":icons/help"))
        self.triggered.connect(lambda: print(len(QApplication.allWindows())))
        parent.addAction(self)
