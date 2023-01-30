from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow
from pydetecdiv.app.gui import MainToolBar, MainStatusBar, FileMenu
from pydetecdiv.app import get_settings


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName('PyDetecDiv main window')

        self.addToolBar(MainToolBar('main toolbar'))

        FileMenu(self)

        self.setStatusBar(MainStatusBar())

        settings = get_settings()
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("windowState"))

    def closeEvent(self, event: QCloseEvent) -> None:
        settings = get_settings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
