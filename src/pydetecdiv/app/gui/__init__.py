#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Main widgets to use with persistent windows
"""
from typing import TYPE_CHECKING
import psutil
import numpy as np

from PySide6.QtCore import QTimer, QRect
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolBar, QStatusBar, QMenu, QApplication, QDialog, QDialogButtonBox, QSizePolicy, QLabel

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui import ActionsSettings, ActionsProject, Enable
import pydetecdiv.app.gui.resources_rc
from pydetecdiv.app.gui.tools import ToolAction
from pydetecdiv.app.gui.tools.data.hdf5 import Create_ROI_HDF5Dialog

if TYPE_CHECKING:
    from pydetecdiv.app.gui.Windows import MainWindow

class FileMenu(QMenu):
    """
    The main window File menu
    """

    def __init__(self, parent: 'MainWindow', *args, **kwargs):
        super().__init__(*args, **kwargs)
        menu = parent.menuBar().addMenu("&File")
        # ActionsProject.OpenProject(menu).setShortcut("Ctrl+O")
        # ActionsProject.NewProject(menu).setShortcut("Ctrl+N")
        # ActionsProject.DeleteProject(menu).setShortcut("Ctrl+D")
        menu.addSeparator()
        ActionsSettings.Settings(menu)
        ActionsSettings.ManageDataSourceAction(menu)
        menu.addSeparator()
        Quit(menu).setShortcut("Ctrl+Q")


class ProjectMenu(QMenu):
    """
        The main window Project menu to manage existing projects
        """

    def __init__(self, parent: 'MainWindow', *args, **kwargs):
        super().__init__(*args, **kwargs)
        menu = parent.menuBar().addMenu("&Project")
        ActionsProject.OpenProject(menu).setShortcut("Ctrl+O")
        ActionsProject.NewProject(menu).setShortcut("Ctrl+N")
        ActionsProject.DeleteProject(menu).setShortcut("Ctrl+D")
        menu.addSeparator()
        configure_source = ActionsProject.ConfigureProjectSourceDir(menu)
        configure_source.setShortcut("Ctrl+C")
        menu.addSeparator()
        upgrade_repository = ActionsProject.UpgradeRepository(menu)
        PyDetecDiv.app.project_selected.connect(lambda _: configure_source.setEnabled(True))
        PyDetecDiv.app.project_selected.connect(lambda _: upgrade_repository.setEnabled(True))


class MainToolBar(QToolBar):
    """
    The main toolbar of the main window
    """

    def __init__(self, parent: 'MainWindow', name: str = 'Main toolbar', **kwargs):
        super().__init__(name, parent=parent, **kwargs)
        self.setObjectName(name)
        ActionsProject.OpenProject(self)
        ActionsProject.NewProject(self)
        ActionsProject.DeleteProject(self)
        Quit(self)
        Help(self)


class MainStatusBar(QStatusBar):
    """
    The status bar of the main window
    """

    def __init__(self):
        super().__init__()
        self.setObjectName('Main status bar')
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_memory_usage)
        self.timer.setInterval(10000)
        self.timer.start()

    def show_memory_usage(self) -> None:
        """
        Show memory usage in status bar
        """
        self.showMessage(
                f'{np.format_float_positional(psutil.Process().memory_info().rss / (1024 * 1024), precision=1)} MB')


class Quit(QAction):
    """
    Quit action, interrupting the application
    """

    def __init__(self, parent: 'MainWindow | MainToolBar | QMenu'):
        super().__init__(QIcon(":icons/exit"), "&Quit", parent)
        self.triggered.connect(QApplication.quit)
        parent.addAction(self)


class Help(QAction):
    """
    Action requesting global help
    """

    def __init__(self, parent: 'MainWindow | MainToolBar'):
        super().__init__(QIcon(":icons/help"), "&Help", parent)
        self.triggered.connect(self.show_info)
        parent.addAction(self)

    def show_info(self) -> None:
        """
        Shows information about the application (version, authors, licence)
        """
        about_dialog = QDialog(self.parent())
        about_dialog.resize(402, 298)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        about_dialog.setSizePolicy(sizePolicy)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, about_dialog)
        button_box.setGeometry(QRect(10, 250, 371, 32))
        button_box.setCenterButtons(True)
        button_box.accepted.connect(about_dialog.close)
        about_dialog.show()

        label = QLabel(about_dialog)
        label.setGeometry(QRect(70, 20, 271, 71))

        label_2 = QLabel(about_dialog)
        label_2.setGeometry(QRect(130, 100, 131, 16))

        label_3 = QLabel(about_dialog)
        label_3.setObjectName("label_3")
        label_3.setGeometry(QRect(20, 130, 361, 50))

        label_4 = QLabel(about_dialog)
        label_4.setGeometry(QRect(80, 190, 231, 20))
        label_5 = QLabel(about_dialog)
        label_5.setGeometry(QRect(130, 220, 131, 16))
        label.setText('<center><h1>pyDetecDiv</h1></center>')
        label_2.setText(f'<center><p>version {PyDetecDiv.version}</p></center>')
        label_3.setText('<center><p>CeCILL FREE SOFTWARE LICENSE AGREEMENT<br/>v2.1 (2013-06-21)</p></center>')
        label_4.setText('<i>https://github.com/fplewniak/pyDetecDiv</i>')
        label_5.setText('<i>f.plewniak@unistra.fr</i>')
        label.show()
        label_2.show()
        label_3.show()
        label_4.show()
        label_5.show()
