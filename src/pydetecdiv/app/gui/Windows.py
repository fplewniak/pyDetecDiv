"""
Classes for persistent windows of the GUI
"""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QAction, QIcon, QPixmap, QImage
from PySide6.QtWidgets import QMainWindow, QMdiArea, QTabWidget, QDockWidget, QFormLayout, QLabel, QComboBox, \
    QDialogButtonBox, QWidget, QFrame, QMenuBar, QVBoxLayout, QPushButton, QHBoxLayout, QGridLayout, QToolButton, \
    QGraphicsView, QGraphicsScene
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from pydetecdiv.app.gui import MainToolBar, MainStatusBar, FileMenu, DataMenu
from pydetecdiv.app import get_settings, PyDetecDiv, pydetecdiv_project, DrawingTools

from pydetecdiv.app.gui.ImageViewer import ImageViewer
import dask.array as da

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

        self.setStatusBar(MainStatusBar())

        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)
        self.image_resource_selector = ImageResourceChooser(self, )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.image_resource_selector, Qt.Vertical)
        self.drawing_tools = DrawingToolsPalette(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.drawing_tools, Qt.Vertical)
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
        if title not in self.tabs:
            self.tabs[title] = TabbedViewer(title, self)
            self.tabs[title].window = self.mdi_area.addSubWindow(self.tabs[title])
            self.tabs[title].setMovable(True)
            self.tabs[title].setTabsClosable(True)
            self.tabs[title].tabCloseRequested.connect(self.tabs[title].close_tab)
            self.tabs[title].show()
        return self.tabs[title]

    def subwindow_activation(self, subwindow):
        if subwindow is not None:
            for c in subwindow.children():
                if (c in self.tabs.values()) and c.viewer.project_name:
                    PyDetecDiv().project_selected.emit(c.viewer.project_name)
                    self.image_resource_selector.position_choice.setCurrentText(c.viewer.fov)
                    self.image_resource_selector.stage_choice.setCurrentText(c.viewer.stage)
                    self.image_resource_selector.channel_choice.setCurrentText(str(c.viewer.C))
                    PyDetecDiv().project_name = c.viewer.project_name


class TabbedViewer(QTabWidget):
    def __init__(self, title, parent=None):
        super().__init__()
        self.viewer = ImageViewer()
        self.setWindowTitle(title)
        self.setDocumentMode(True)
        self.addTab(self.viewer, 'FOV')
        self.parent = parent
        self.window = None

    def closeEvent(self, event):
        del (self.parent.tabs[self.windowTitle()])

    def show_plot(self, df, title='Plot'):
        plot_viewer = MatplotViewer(self)
        self.addTab(plot_viewer, title)
        df.plot(ax=plot_viewer.axes)
        plot_viewer.canvas.draw()
        self.setCurrentWidget(plot_viewer)

    def show_image(self, data, title='Image'):
        viewer = QGraphicsView(self)
        scene = QGraphicsScene()
        pixmap = QPixmap()
        pixmapItem = scene.addPixmap(pixmap)
        ny, nx = data.shape
        img = QImage(np.ascontiguousarray(data), nx, ny, QImage.Format_Grayscale16)
        pixmap.convertFromImage(img)
        pixmapItem.setPixmap(pixmap)
        viewer.setScene(scene)
        self.addTab(viewer, title)

    def close_tab(self, index):
        if self.widget(index) != self.viewer:
            self.removeTab(index)


class MatplotViewer(QWidget):
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
    def __init__(self, parent):
        super().__init__('Image resource selector', parent)
        self.setObjectName('Image_resource_selector')
        self.form = QFrame()
        self.form.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        self.formLayout = QFormLayout(self.form)
        self.formLayout.setObjectName(u"formLayout")

        self.position_label = QLabel('Position', self.form)
        self.position_label.setObjectName(u"position_label")
        self.position_choice = QComboBox(self.form)
        self.position_choice.setObjectName(u"position_choice")
        self.formLayout.addRow(self.position_label, self.position_choice)

        self.stage_label = QLabel('Stage', self.form)
        self.stage_label.setObjectName(u"stage_label")
        self.stage_choice = QComboBox(self.form)
        self.stage_choice.setObjectName(u"stage_choice")
        self.formLayout.addRow(self.stage_label, self.stage_choice)

        self.channel_label = QLabel('Channel', self.form)
        self.channel_label.setObjectName(u"channel_label")
        self.channel_choice = QComboBox(self.form)
        self.channel_choice.setObjectName(u"channel_choice")
        self.formLayout.addRow(self.channel_label, self.channel_choice)

        self.OK_button = QDialogButtonBox(self.form)
        self.OK_button.setObjectName(u"OK_button")
        self.OK_button.setStandardButtons(QDialogButtonBox.Ok)
        self.formLayout.addWidget(self.OK_button)

        self.setWidget(self.form)

        PyDetecDiv().project_selected.connect(self.set_choice)
        self.OK_button.accepted.connect(self.accept)
        self.position_choice.currentIndexChanged.connect(self.update_channel_choice)
        self.stage_choice.currentIndexChanged.connect(self.update_channel_choice)

    def set_choice(self, p_name):
        with pydetecdiv_project(p_name) as project:
            self.position_choice.clear()
            self.stage_choice.clear()
            self.channel_choice.clear()
            if project.count_objects('FOV'):
                FOV_list = [fov.name for fov in project.get_objects('FOV')]
                dataset_list = [ds.name for ds in project.get_objects('Dataset')]
                self.position_choice.addItems(sorted(FOV_list))
                self.stage_choice.addItems(dataset_list)
                fov = project.get_named_object('FOV', self.position_choice.currentText())
                dataset = self.stage_choice.currentText()
                self.channel_choice.addItems([str(i) for i in range(fov.image_resource(dataset).sizeC)])

    def update_channel_choice(self):
        if self.stage_choice.currentText() and self.position_choice.currentText():
            with pydetecdiv_project(PyDetecDiv().project_name) as project:
                fov = project.get_named_object('FOV', self.position_choice.currentText())
                dataset = self.stage_choice.currentText()
                self.channel_choice.clear()
                self.channel_choice.addItems([str(i) for i in range(fov.image_resource(dataset).sizeC)])

    def accept(self):
        PyDetecDiv().setOverrideCursor(QCursor(Qt.WaitCursor))
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            fov = project.get_named_object('FOV', self.position_choice.currentText())
            dataset = self.stage_choice.currentText()
            image_resource = fov.image_resource(dataset)
        tab_key = f'{PyDetecDiv().project_name}/{fov.name}/{dataset}'
        tab = self.parent().add_tabbbed_viewer(tab_key)
        tab.setWindowTitle(tab_key)
        tab.viewer.set_image_resource(image_resource)
        tab.viewer.set_channel(self.channel_choice.currentIndex())
        tab.viewer.display()
        tab.viewer.project_name = PyDetecDiv().project_name
        tab.viewer.fov = fov.name
        tab.viewer.stage = dataset
        PyDetecDiv().setOverrideCursor(QCursor(Qt.ArrowCursor))


class DrawingToolsPalette(QDockWidget):
    def __init__(self, parent):
        super().__init__('Drawing tools', parent)
        self.setObjectName('Drawing_tools_palette')
        self.form = QFrame()
        self.form.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        self.formLayout = QGridLayout(self.form)
        self.formLayout.setObjectName(u"drawingToolsLayout")

        self.cursor_button = Cursor(self)
        self.draw_ROI_button = DrawROI(self)
        self.create_ROIs_button = DuplicateROI(self)
        self.tools = [self.cursor_button, self.draw_ROI_button, self.create_ROIs_button]

        self.formLayout.addWidget(self.cursor_button, 0, 0)
        self.formLayout.addWidget(self.draw_ROI_button, 0, 1)
        self.formLayout.addWidget(self.create_ROIs_button, 0, 2)

        self.form.setLayout(self.formLayout)

        self.setWidget(self.form)

    def unset_tools(self):
        for t in self.tools:
            t.setChecked(False)

    def current_tool(self):
        for t in self.tools:
            if t.isChecked():
                return t


class Cursor(QToolButton):
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
        self.parent.unset_tools()
        self.setChecked(True)
        PyDetecDiv().current_drawing_tool = DrawingTools.Cursor


class DrawROI(QToolButton):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setIcon(QIcon(":icons/draw_ROI"))
        self.setToolTip(DrawingTools.DrawROI)
        self.setCheckable(True)
        self.clicked.connect(self.select_tool)

    def select_tool(self):
        self.parent.unset_tools()
        self.setChecked(True)
        PyDetecDiv().current_drawing_tool = DrawingTools.DrawROI


class DuplicateROI(QToolButton):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setIcon(QIcon(":icons/duplicate_ROI"))
        self.setToolTip(DrawingTools.DuplicateROI)
        self.setCheckable(True)
        self.clicked.connect(self.select_tool)

    def select_tool(self):
        self.parent.unset_tools()
        self.setChecked(True)
        PyDetecDiv().current_drawing_tool = DrawingTools.DuplicateROI
