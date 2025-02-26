#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Definition of widgets to display Tabs in a Tabbed window
"""
import random

import pandas
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QTabWidget, QWidget

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers.plots import MatplotViewer


class TabbedWindow(QTabWidget):
    """
    A class for tabbed windows. These windows are displayed in the MDI area of the main window.
    """
    def __init__(self, title: str):
        super().__init__()
        self.project_name = None
        self.top_widget = None
        self.setWindowTitle(title)
        self.setDocumentMode(True)

        self.window = PyDetecDiv.main_window.mdi_area.addSubWindow(self)
        mdi_space = PyDetecDiv.main_window.mdi_area.geometry()
        xmax, ymax = mdi_space.width() * 0.20, mdi_space.height() * 0.20
        x, y = random.uniform(0, xmax), random.uniform(0, ymax)
        self.window.setGeometry(x, y, mdi_space.width() * 0.8, mdi_space.height() * 0.8)
        self.setMovable(True)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)
        self.show()

    def set_top_tab(self, widget: QWidget, title: str) -> None:
        """
        Set the main tab for this tabbed window, i.e. the reference tab that cannot be closed
        :param widget: the widget to display in the top tab
        :param title: the title of the top tab
        """
        if self.top_widget is None:
            self.top_widget = self.widget(self.addTab(widget, title))
        else:
            new_tab = self.widget(self.addTab(widget, title))
            self.setCurrentWidget(new_tab)

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Close the current tabbed widget window

        :param event: the close event
        :type event: QCloseEvent
        """
        for i in range(PyDetecDiv.main_window.tabs[self.windowTitle()].count()):
            QCoreApplication.sendEvent(PyDetecDiv.main_window.tabs[self.windowTitle()].widget(i), QCloseEvent())
        del PyDetecDiv.main_window.tabs[self.windowTitle()]
        PyDetecDiv.main_window.scene_tree_palette.reset()

    def close_tab(self, index: int) -> None:
        """
        Close the tab with the specified index

        :param index: the index of the tab to close
        :type index: int
        """
        if self.widget(index) != self.top_widget:
            self.removeTab(index)

    def show_plot(self, df: pandas.DataFrame, title: str = 'Plot') -> None:
        """
        Open a viewer tab to plot a graphic from a pandas dataframe

        :param df: the data to plot
        :type df: pandas DataFrame
        :param title: the title for the plot tab
        :type title: str
        """
        plot_viewer = MatplotViewer(self)
        self.addTab(plot_viewer, title)
        df.plot(ax=plot_viewer.axes)
        plot_viewer.show()
        self.setCurrentWidget(plot_viewer)
