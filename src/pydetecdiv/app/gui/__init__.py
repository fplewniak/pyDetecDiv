#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
from PySide6.QtWidgets import QToolBar, QStatusBar, QMenu
import pydetecdiv.app.gui.Actions


class FileMenu(QMenu):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        menu = parent.menuBar().addMenu("&File")
        Actions.OpenProject(menu, icon=True).setShortcut("Ctrl+O")
        Actions.Settings(menu)
        menu.addSeparator()
        Actions.Quit(menu).setShortcut("Ctrl+Q")


class MainToolBar(QToolBar):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName('Main toolbar')

        Actions.OpenProject(self, icon=True)
        Actions.Settings(self, icon=True)
        Actions.Quit(self, icon=True)


class MainStatusBar(QStatusBar):
    def __init__(self):
        super().__init__()
        self.setObjectName('Main status bar')


