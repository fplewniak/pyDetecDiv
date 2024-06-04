import random

from PySide6.QtWidgets import QTabWidget

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers.plots import MatplotViewer


class TabbedWindow(QTabWidget):
    def __init__(self, title):
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

    def set_top_tab(self, widget, title):
        self.top_widget = self.widget(self.addTab(widget, title))

    def closeEvent(self, _):
        """
        Close the current tabbed widget window

        :param event: the close event
        :type event: QCloseEvent
        """
        del PyDetecDiv.main_window.tabs[self.windowTitle()]

    def close_tab(self, index):
        """
        Close the tab with the specified index

        :param index: the index of the tab to close
        :type index: int
        """
        if self.widget(index) != self.top_widget:
            self.removeTab(index)

    def show_plot(self, df, title='Plot'):
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
