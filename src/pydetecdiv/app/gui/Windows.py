"""
Classes for persistent windows of the GUI
"""
from PySide6.QtWidgets import QMainWindow
from pydetecdiv.app.gui import MainToolBar, MainStatusBar, FileMenu
from pydetecdiv.app import PyDetecDivApplication, get_settings


class MainWindow(QMainWindow):
    """
    The principal window
    """

    def __init__(self):
        super().__init__()
        PyDetecDivApplication.main_window = self
        self.setObjectName('PyDetecDiv main window')

        self.addToolBar(MainToolBar('main toolbar'))

        FileMenu(self)

        self.setStatusBar(MainStatusBar())

        settings = get_settings()
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("windowState"))

    def closeEvent(self, _):
        """
        Response to close event signal. Settings are saved in order to save the current window geometry and state.
        :param event: the event object
        :type event: QCloseEvent
        """
        settings = get_settings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
