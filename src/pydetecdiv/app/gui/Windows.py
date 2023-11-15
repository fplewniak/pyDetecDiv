"""
Classes for persistent windows of the GUI
"""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QIcon, QPixmap, QImage
from PySide6.QtWidgets import QMainWindow, QMdiArea, QTabWidget, QDockWidget, QFormLayout, QLabel, QComboBox, \
    QDialogButtonBox, QWidget, QFrame, QVBoxLayout, QGridLayout, QToolButton, \
    QGraphicsView, QGraphicsScene, QSpinBox, QGroupBox, QHBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from pydetecdiv.app.gui import MainToolBar, MainStatusBar, FileMenu, DataMenu, PluginMenu
from pydetecdiv.app import get_settings, PyDetecDiv, pydetecdiv_project, DrawingTools

from pydetecdiv.app.gui.ImageViewer import ImageViewer
from pydetecdiv.app.gui.Toolbox import ToolboxTreeView, ToolboxTreeModel


class MainWindow(QMainWindow):
    """
    The principal window
    """

    def __init__(self):
        super().__init__()
        self.setObjectName('PyDetecDiv main window')

        self.addToolBar(MainToolBar('main toolbar'))

        self.tabs = {}

        FileMenu(self)
        DataMenu(self)
        PluginMenu(self)

        self.setStatusBar(MainStatusBar())

        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)
        self.image_resource_selector = ImageResourceChooser(self, )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.image_resource_selector, Qt.Vertical)
        self.drawing_tools = DrawingToolsPalette(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.drawing_tools, Qt.Vertical)
        self.analysis_tools = AnalysisToolsTree(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.analysis_tools, Qt.Vertical)
        self.mdi_area.subWindowActivated.connect(self.subwindow_activation)
        PyDetecDiv().project_selected.connect(self.setWindowTitle)

        settings = get_settings()
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("windowState"))

        self.current_tool = None

    def closeEvent(self, _):
        """
        Response to close event signal. Settings are saved in order to save the current window geometry and state.

        :param event: the event object
        :type event: QCloseEvent
        """
        settings = get_settings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def add_tabbbed_viewer(self, title):
        """
        Add a new Tabbed viewer to visualize a FOV and its related information and analyses

        :param title: the title for the tabbed viewer window (i.e. Project/FOV/dataset
        :type title: str
        :return: the new tabbed viewer widget
        :rtype: TabbedViewer
        """
        if title not in self.tabs:
            self.tabs[title] = TabbedViewer(title, self)
            self.tabs[title].window = self.mdi_area.addSubWindow(self.tabs[title])
            self.tabs[title].setMovable(True)
            self.tabs[title].setTabsClosable(True)
            self.tabs[title].tabCloseRequested.connect(self.tabs[title].close_tab)
            self.tabs[title].show()
        return self.tabs[title]

    def subwindow_activation(self, subwindow):
        """
        When a tabbed viewer is activated (its focus is set), then the Image resource selector should be fed with the
        corresponding image resource information(FOV name, stage dataset, channel)

        :param subwindow: the activated sub-window
        :type subwindow:QMdiArea
        """
        if subwindow is not None:
            for c in subwindow.children():
                if (c in self.tabs.values()) and c.viewer.project_name:
                    PyDetecDiv().project_selected.emit(c.viewer.project_name)
                    self.image_resource_selector.position_choice.setCurrentText(c.viewer.fov)
                    # self.image_resource_selector.stage_choice.setCurrentText(c.viewer.stage)
                    # self.image_resource_selector.channel_choice.setCurrentText(str(c.viewer.C))
                    PyDetecDiv().project_name = c.viewer.project_name

    @property
    def active_subwindow(self):
        active_subwindow = self.mdi_area.activeSubWindow()
        if active_subwindow:
            return [tab for tab in PyDetecDiv().main_window.tabs.values() if tab.window == active_subwindow][0]
        return None


class TabbedViewer(QTabWidget):
    """
    A tabbed widget to hold the FOV main viewer and all related viewers (plots, image resources, etc.)
    """

    def __init__(self, title, parent=None):
        super().__init__()
        self.viewer = ImageViewer()
        self.setWindowTitle(title)
        self.setDocumentMode(True)
        self.addTab(self.viewer, 'FOV')
        self.parent = parent
        self.window = None
        self.drift = None

    def closeEvent(self, _):
        """
        Close the current tabbed widget window

        :param event: the close event
        :type event: QCloseEvent
        """
        del self.parent.tabs[self.windowTitle()]

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
        plot_viewer.canvas.draw()
        self.setCurrentWidget(plot_viewer)

    def show_image(self, data, title='Image', format_=QImage.Format_Grayscale16):
        """
        Display a 2D image

        :param data: the 2D image data
        :type data: ndarray
        :param title: the title of the tab
        :type title: str
        :param format: the image format
        :type format: QImage.Format
        """
        viewer = QGraphicsView(self)
        scene = QGraphicsScene()
        pixmap = QPixmap()
        pixmapItem = scene.addPixmap(pixmap)
        match format_:
            case QImage.Format_Grayscale16|QImage.Format_Grayscale8:
                print('Grayscale')
                ny, nx = data.shape
                img = QImage(np.ascontiguousarray(data), nx, ny, format_)
            case QImage.Format_RGB888:
                print('RGB888')
                ny, nx, nc = data.shape
                img = QImage(np.ascontiguousarray(data), nx, ny, nc*nx, format_)
            case _:
                ...
        pixmap.convertFromImage(img)
        pixmapItem.setPixmap(pixmap)
        viewer.setScene(scene)
        self.addTab(viewer, title)
        self.setCurrentWidget(viewer)

    def close_tab(self, index):
        """
        Close the tab with the specified index

        :param index: the index of the tab to close
        :type index: int
        """
        if self.widget(index) != self.viewer:
            self.removeTab(index)

    def get_image_viewers(self):
        """
        Get the list of image viewers in the current Tabbed viewer

        :return: the list of image viewers
        :rtype: list of ImageViewer widgets
        """
        return [self.widget(i) for i in range(self.count()) if isinstance(self.widget(i), ImageViewer)]


class MatplotViewer(QWidget):
    """
    A widget to display matplotlib plots in a tab
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.dismiss_button = QPushButton('Dismiss')
        # self.dismiss_button.clicked.connect(lambda: self.parent().removeWidget(self))
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.axes = self.canvas.figure.subplots()
        self.canvas.figure.tight_layout(pad=0.1)
        self.toolbar = QWidget(self)
        self.matplot_toolbar = NavigationToolbar(self.canvas, self)

        # hlayout = QHBoxLayout()
        # hlayout.addWidget(self.matplot_toolbar)
        # hlayout.addWidget(self.dismiss_button)
        # self.toolbar.setLayout(hlayout)

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas)
        vlayout.addWidget(self.matplot_toolbar)
        # vlayout.addWidget(self.toolbar)
        self.setLayout(vlayout)


class ImageResourceChooser(QDockWidget):
    """
    A dockable widget with a form for choosing an image resource to display in a new tabbed viewer. The image resource
    is determined by the FOV name, the dataset (stage) and a channel
    """

    def __init__(self, parent):
        super().__init__('Image resource selector', parent)
        self.setObjectName('Image_resource_selector')
        self.form = QFrame()
        self.form.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        self.formLayout = QFormLayout(self.form)
        self.formLayout.setObjectName("formLayout")

        self.position_label = QLabel('Position', self.form)
        self.position_label.setObjectName("position_label")
        self.position_choice = QComboBox(self.form)
        self.position_choice.setObjectName("position_choice")
        self.formLayout.addRow(self.position_label, self.position_choice)

        self.OK_button = QDialogButtonBox(self.form)
        self.OK_button.setObjectName("run_button")
        self.OK_button.setStandardButtons(QDialogButtonBox.Ok)
        self.formLayout.addWidget(self.OK_button)

        self.setWidget(self.form)

        PyDetecDiv().project_selected.connect(self.set_choice)
        self.OK_button.accepted.connect(self.accept)

    def set_choice(self, p_name):
        """
        Set the available values for FOVs, datasets and channels given a project name

        :param p_name: the project name
        :type p_name: str
        """
        with pydetecdiv_project(p_name) as project:
            self.position_choice.clear()
            if project.count_objects('FOV'):
                self.position_choice.addItems(sorted([fov.name for fov in project.get_objects('FOV')]))

    def accept(self):
        """
        When the OK button is clicked, then open a new TabbedViewer window and display the selected Image resource
        """
        PyDetecDiv().setOverrideCursor(QCursor(Qt.WaitCursor))
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            fov = project.get_named_object('FOV', self.position_choice.currentText())
            roi_list = fov.roi_list
            image_resource = fov.image_resource('data').image_resource_data()
        tab_key = f'{PyDetecDiv().project_name}/{fov.name}'
        tab = self.parent().add_tabbbed_viewer(tab_key)
        tab.setWindowTitle(tab_key)
        tab.viewer.set_image_resource_data(image_resource)
        tab.viewer.set_channel(0)
        tab.viewer.display()
        tab.viewer.draw_saved_rois(roi_list)
        tab.viewer.project_name = PyDetecDiv().project_name
        tab.viewer.fov = fov.name
        tab.viewer.stage = 'data'
        PyDetecDiv().restoreOverrideCursor()


class DrawingToolsPalette(QDockWidget):
    """
    A dockable window with tools for drawing ROIs and other items.
    """

    def __init__(self, parent):
        super().__init__('Drawing tools', parent)
        self.setObjectName('Drawing_tools_palette')
        self.palette = QFrame()
        self.palette_layout = QVBoxLayout(self.palette)

        self.form = QFrame()
        self.form.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        self.formLayout = QGridLayout(self.form)
        self.formLayout.setObjectName("drawingToolsLayout")

        self.cursor_button = Cursor(self)
        self.draw_ROI_button = DrawROI(self)
        self.create_ROIs_button = DuplicateROI(self)
        self.tools = [self.cursor_button, self.draw_ROI_button, self.create_ROIs_button]

        self.formLayout.addWidget(self.cursor_button, 0, 0)
        self.formLayout.addWidget(self.draw_ROI_button, 0, 1)
        self.formLayout.addWidget(self.create_ROIs_button, 0, 2)

        self.form.setLayout(self.formLayout)

        self.properties = QFrame()
        self.properties.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        self.properties_layout = QVBoxLayout(self.properties)
        self.roi_prop_box = QGroupBox(self)
        self.roi_width = QSpinBox(self)
        self.roi_width.setMaximum(9999)
        self.roi_width.setMinimum(5)
        self.roi_prop_box_layout = QHBoxLayout(self.roi_prop_box)
        label = QLabel(self)
        label.setText('width:')
        self.roi_prop_box_layout.addWidget(label)
        self.roi_prop_box_layout.addWidget(self.roi_width)
        self.roi_height = QSpinBox(self)
        self.roi_height.setMaximum(9999)
        self.roi_height.setMinimum(5)
        label = QLabel(self)
        label.setText('height:')
        self.roi_prop_box_layout.addWidget(label)
        self.roi_prop_box_layout.addWidget(self.roi_height)
        self.properties_layout.addWidget(self.roi_prop_box)

        self.roi_width.valueChanged.connect(self.set_ROI_width)
        self.roi_height.valueChanged.connect(self.set_ROI_height)

        self.palette_layout.addWidget(self.form)
        self.palette_layout.addWidget(self.properties)
        self.setWidget(self.palette)

    def unset_tools(self):
        """
        Unset all available tools
        """
        for t in self.tools:
            t.setChecked(False)

    def current_tool(self):
        """
        Return the currently checked tool

        :return: the currently checked tool
        :rtype: QToolButton
        """
        for t in self.tools:
            if t.isChecked():
                return t
        return None

    def set_ROI_width(self, width):
        PyDetecDiv().main_window.active_subwindow.viewer.scene.set_ROI_width(width)

    def set_ROI_height(self, height):
        PyDetecDiv().main_window.active_subwindow.viewer.scene.set_ROI_height(height)

class Cursor(QToolButton):
    """
    QToolButton to activate the tool for selecting and dragging items in the view
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setIcon(QIcon(":icons/cursor"))
        self.setToolTip(DrawingTools.Cursor)
        self.setCheckable(True)
        self.clicked.connect(self.select_tool)
        self.setChecked(True)
        PyDetecDiv().current_drawing_tool = DrawingTools.Cursor

    def select_tool(self):
        """
        Select the Cursor tool
        """
        self.parent.unset_tools()
        self.setChecked(True)
        PyDetecDiv().current_drawing_tool = DrawingTools.Cursor


class DrawROI(QToolButton):
    """
    A QToolButton to activate the tool for drawing a ROI
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setIcon(QIcon(":icons/draw_ROI"))
        self.setToolTip(DrawingTools.DrawROI)
        self.setCheckable(True)
        self.clicked.connect(self.select_tool)

    def select_tool(self):
        """
        Select the DrawROI tool
        """
        self.parent.unset_tools()
        self.setChecked(True)
        PyDetecDiv().current_drawing_tool = DrawingTools.DrawROI


class DuplicateROI(QToolButton):
    """
    A QToolButton to activate the tool for duplicating a ROI
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setIcon(QIcon(":icons/duplicate_ROI"))
        self.setToolTip(DrawingTools.DuplicateROI)
        self.setCheckable(True)
        self.clicked.connect(self.select_tool)

    def select_tool(self):
        """
        Select the DuplicateROI tool
        """
        self.parent.unset_tools()
        self.setChecked(True)
        PyDetecDiv().current_drawing_tool = DrawingTools.DuplicateROI


class AnalysisToolsTree(QDockWidget):
    """
    A dockable window with tools for image analysis.
    """

    def __init__(self, parent):
        super().__init__('Analysis tools', parent)
        self.setObjectName('Analysis_tools_tree')
        tree_view = ToolboxTreeView()
        tree_view.setModel(ToolboxTreeModel(parent=self))
        self.setWidget(tree_view)
