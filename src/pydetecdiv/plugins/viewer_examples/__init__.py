"""
A showcase plugin showing how to interact with TabbedViewer and ImageViewer objects
"""
import pandas
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QAction

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv
from pydetecdiv.plugins.viewer_examples.gui import AddPlotDialog


class Plugin(plugins.Plugin):
    """
    A class extending plugins.Plugin to handle the showcase plugin
    """
    id_ = 'gmgm.plewniak.viewer.addons'
    version = '1.0.0'
    name = 'Viewer add-ons'
    category = 'Plugin examples'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def addActions(self, menu):
        """
        Create a submenu and add actions thereto
        :param menu: the parent menu
        :type menu: QMenu
        """
        submenu = menu.addMenu(self.name)
        action_launch = QAction("Plot dialog window", submenu)
        action_launch.triggered.connect(self.launch)
        submenu.addAction(action_launch)

        action_plot = QAction("Add plot", submenu)
        action_plot.triggered.connect(self.add_plot)
        submenu.addAction(action_plot)

        action_change_pen = QAction("change pen", submenu)
        action_change_pen.triggered.connect(self.change_pen)
        submenu.addAction(action_change_pen)

    def launch(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self.gui = AddPlotDialog(PyDetecDiv().main_window)
        self.gui.button_box.accepted.connect(self.add_plot)
        self.gui.button_box.accepted.connect(self.gui.close)
        self.gui.exec()

    def change_pen(self):
        """
        Toggle the pen style (colour and width) for drawing regions in the current subwindow
        """
        active_subwindow = PyDetecDiv().main_window.mdi_area.activeSubWindow()
        if active_subwindow:
            tab = [tab for tab in PyDetecDiv().main_window.tabs.values() if tab.window == active_subwindow][0]
            if tab.viewer.scene.pen.width() == 2:
                tab.viewer.scene.pen = QPen(Qt.GlobalColor.blue, 6)
            else:
                tab.viewer.scene.pen = QPen(Qt.GlobalColor.cyan, 2)

    def add_plot(self):
        """
        Add a new tab with a dummy plot to the currently active subwindow
        """
        active_subwindow = PyDetecDiv().main_window.mdi_area.activeSubWindow()
        if active_subwindow:
            tab = [tab for tab in PyDetecDiv().main_window.tabs.values() if tab.window == active_subwindow][0]
            x = np.linspace(0, 10, 500)
            y = np.sin(x)
            df = pandas.DataFrame(y)
            tab.show_plot(df, 'Plugin plot')
