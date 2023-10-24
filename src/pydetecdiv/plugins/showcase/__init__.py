"""
A showcase plugin showing how to interact with TabbedViewer and ImageViewer objects
"""
import pandas
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv
from pydetecdiv.plugins.showcase.Actions import Action1, Action2, Action3

class Plugin(plugins.Plugin):
    """
    A class extending plugins.Plugin to handle the showcase plugin
    """
    id_ = 'gmgm.plewniak.viewer.addons'
    version = '1.0.0'
    name = 'Viewer add-ons'
    category = 'Showcase'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def addActions(self, menu):
        """
        Add actions to the submenu
        :param menu: the Showcase submenu
        :type menu: QMenu
        """
        Action1(menu, self)
        Action3(menu, self)
        Action2(menu, self)

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
