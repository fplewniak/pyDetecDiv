"""
Classes for persistent windows of the GUI
"""
from PySide6.QtWidgets import QMainWindow, QMdiArea
from pydetecdiv.app.gui import MainToolBar, MainStatusBar, FileMenu, DataMenu
from pydetecdiv.app import get_settings

from pydetecdiv.app.gui.ImageViewer import ImageViewer

class MainWindow(QMainWindow):
    """
    The principal window
    """

    def __init__(self):
        super().__init__()
        self.setObjectName('PyDetecDiv main window')

        self.addToolBar(MainToolBar('main toolbar'))

        FileMenu(self)
        DataMenu(self)

        self.setStatusBar(MainStatusBar())

        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)
        self.viewer = ImageViewer()
        self.mdi_area.addSubWindow(self.viewer)
        self.viewer.showMaximized()

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
