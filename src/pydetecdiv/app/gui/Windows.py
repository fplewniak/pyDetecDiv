from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow
from pydetecdiv.app.gui import MainToolBar, MainStatusBar, FileMenu
import pydetecdiv.app


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        pydetecdiv.app.main_window = self
        self.setObjectName('PyDetecDiv main window')

        self.addToolBar(MainToolBar(self, 'main toolbar'))

        FileMenu(self)

        self.setStatusBar(MainStatusBar())

        settings = pydetecdiv.app.get_settings()
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("windowState"))

    def closeEvent(self, event: QCloseEvent) -> None:
        settings = pydetecdiv.app.get_settings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
