"""
Classes for persistent windows of the GUI
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QIcon
from PySide6.QtWidgets import QMainWindow, QMdiArea, QDockWidget, QLabel, QComboBox, \
    QDialogButtonBox, QFrame, QVBoxLayout, QGridLayout, QToolButton, QSpinBox, QGroupBox, QHBoxLayout, QCheckBox

from pydetecdiv.app.gui import MainToolBar, MainStatusBar, FileMenu, DataMenu, PluginMenu
from pydetecdiv.app import get_settings, PyDetecDiv, pydetecdiv_project, DrawingTools
from pydetecdiv.app.gui.FOVmanager import FOVmanager

from pydetecdiv.app.gui.Toolbox import ToolboxTreeView, ToolboxTreeModel
from pydetecdiv.app.gui.core.widgets.palettes.objects import ObjectTreePalette
from pydetecdiv.app.gui.core.widgets.TabWidgets import TabbedWindow


class MainWindow(QMainWindow):
    """
    The principal window
    """

    def __init__(self):
        super().__init__()
        self.setObjectName('PyDetecDiv main window')

        self.addToolBar(MainToolBar(self, 'Main Toolbar'))

        self.tabs = {}

        FileMenu(self)
        DataMenu(self)
        PluginMenu(self)

        self.setStatusBar(MainStatusBar())

        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)
        self.image_resource_selector = ImageResourceChooser(self, )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.image_resource_selector, Qt.Orientation.Vertical)
        self.drawing_tools = DrawingToolsPalette(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.drawing_tools, Qt.Orientation.Vertical)
        self.analysis_tools = AnalysisToolsTree(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_tools, Qt.Orientation.Vertical)
        self.analysis_tools.hide()
        self.object_tree_palette = ObjectTreePalette(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.object_tree_palette, Qt.Orientation.Vertical)
        self.mdi_area.subWindowActivated.connect(self.subwindow_activation)
        PyDetecDiv.app.project_selected.connect(self.setWindowTitle)

        settings = get_settings()
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("windowState"))

        self.current_tool = None

    def closeEvent(self, _):
        """
        Response to close event signal. Settings are saved in order to save the current window geometry and state.
        """
        settings = get_settings()
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    # def add_tabbed_viewer(self, title):
    #     """
    #     Add a new Tabbed viewer to visualize a FOV and its related information and analyses
    #
    #     :param title: the title for the tabbed viewer window (i.e. Project/FOV/dataset
    #     :type title: str
    #     :return: the new tabbed viewer widget
    #     :rtype: TabbedViewer
    #     """
    #     if title not in self.tabs:
    #         self.tabs[title] = TabbedWindow(title)
    #         self.tabs[title].set_top_tab(ImageViewer(), title)
    #     return self.tabs[title]

    def add_tabbed_window(self, title: str) -> TabbedWindow:
        """
        Add a new Tabbed Mdi subwindow to visualize related information and analyses

        :param title: the title for the tabbed viewer window
        :return: the new tabbed viewer widget
        """
        if title not in self.tabs:
            self.tabs[title] = TabbedWindow(title)
        return self.tabs[title]

    def subwindow_activation(self, subwindow: QMdiArea) -> None:
        """
        When a tabbed viewer is activated (its focus is set), then the Image resource selector should be fed with the
        corresponding image resource information(FOV name, stage dataset, channel)

        :param subwindow: the activated sub-window
        """
        if subwindow is not None:
            for c in subwindow.children():
                if (c in self.tabs.values()) and hasattr(c, 'project_name') and c.project_name:
                    PyDetecDiv.app.project_selected.emit(c.project_name)
                    PyDetecDiv.project_name = c.project_name
                    if hasattr(c.top_widget, 'fov'):
                        self.image_resource_selector.position_choice.setCurrentText(c.top_widget.fov)

    @property
    def active_subwindow(self) -> QMdiArea | None:
        """
        A property returning the currently active subwindow in the MDI area
        :return: the currently active tabbed window in the MDI area
        """
        active_subwindow = self.mdi_area.activeSubWindow()
        if active_subwindow:
            return [tab for tab in PyDetecDiv.main_window.tabs.values() if tab.window == active_subwindow][0]
        return None


class ImageResourceChooser(QDockWidget):
    """
    A dockable widget with a form for choosing an image resource to display in a new tabbed viewer. The image resource
    is determined by the FOV name, the dataset (stage) and a channel
    """

    def __init__(self, parent: MainWindow):
        super().__init__('Image resource selector', parent)
        self.setObjectName('Image_resource_selector')
        self.form = QFrame()
        self.form.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)

        layout = QGridLayout(self.form)
        layout.addWidget(QLabel('Position', self.form), 0, 0)
        self.position_choice = QComboBox(self.form)
        layout.addWidget(self.position_choice, 0, 1, 1, 2)

        self.bright_field = QCheckBox('Bright field image', self.form)
        self.bright_field_C = QComboBox(self.form)
        self.bright_field_Z = QComboBox(self.form)

        layout.addWidget(self.bright_field, 1, 0)
        layout.addWidget(QLabel('C'), 1, 1)
        layout.addWidget(self.bright_field_C, 1, 2)
        layout.addWidget(QLabel('Z'), 2, 1)
        layout.addWidget(self.bright_field_Z, 2, 2)

        self.fluorescence = QCheckBox('Fluorescence image', self.form)
        self.fluo_red = QComboBox(self.form)
        self.fluo_Z = QComboBox(self.form)
        self.fluo_green = QComboBox(self.form)
        self.fluo_blue = QComboBox(self.form)
        layout.addWidget(self.fluorescence, 3, 0)
        layout.addWidget(QLabel('R'), 3, 1)
        layout.addWidget(self.fluo_red, 3, 2)
        layout.addWidget(QLabel('G'), 4, 1)
        layout.addWidget(self.fluo_green, 4, 2)
        layout.addWidget(QLabel('B'), 5, 1)
        layout.addWidget(self.fluo_blue, 5, 2)
        layout.addWidget(QLabel('Z'), 6, 1)
        layout.addWidget(self.fluo_Z, 6, 2)

        self.OK_button = QDialogButtonBox(self.form)
        self.OK_button.setStandardButtons(QDialogButtonBox.StandardButton.Ok)
        layout.addWidget(self.OK_button, 7, 2)

        self.form.setLayout(layout)
        self.setWidget(self.form)

        PyDetecDiv.app.project_selected.connect(self.set_choice)
        self.OK_button.accepted.connect(self.accept)

    def set_choice(self, p_name: str) -> None:
        """
        Set the available values for FOVs, datasets and channels given a project name

        :param p_name: the project name
        :type p_name: str
        """
        with pydetecdiv_project(p_name) as project:
            self.position_choice.clear()
            self.bright_field_C.clear()
            self.bright_field_Z.clear()
            self.fluo_red.clear()
            self.fluo_green.clear()
            self.fluo_blue.clear()
            self.fluo_Z.clear()
            if project.count_objects('FOV'):
                self.position_choice.addItems(sorted([fov.name for fov in project.get_objects('FOV')]))
                fov = project.get_object('FOV', 1)
                kval = fov.image_resource().key_val
                if (kval is not None) and ('channel_names' in kval):
                    channel_list = [kval['channel_names'][c] for c in range(fov.image_resource().sizeC)]
                else:
                    channel_list = [str(c) for c in range(fov.image_resource().sizeC)]
                stack_list = [str(z) for z in range(fov.image_resource().sizeZ)]
                self.bright_field_C.addItems(channel_list)
                self.bright_field_Z.addItems(stack_list)
                self.fluo_red.addItems(['n.a'] + channel_list)
                self.fluo_green.addItems(['n.a'] + channel_list)
                self.fluo_blue.addItems(['n.a'] + channel_list)
                self.fluo_Z.addItems(stack_list)

    def accept(self) -> None:
        """
        When the OK button is clicked, then open a new TabbedViewer window and display the selected Image resource
        """
        PyDetecDiv.app.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            fov = project.get_named_object('FOV', self.position_choice.currentText())
            roi_list = fov.roi_list
            image_resource = fov.image_resource('data')
            image_resource_data = image_resource.image_resource_data()

        tab_key = f'{PyDetecDiv.project_name}/{fov.name}'
        tab = PyDetecDiv.main_window.add_tabbed_window(tab_key)
        last_widget = tab.currentWidget()
        tab.set_top_tab(FOVmanager(fov=fov), 'FOV')
        current_widget = tab.currentWidget()
        current_widget.tscale = image_resource.tscale * image_resource.tunit
        # frame = last_widget.T if tab.count() > 1 else 0
        frame=0
        C_bright_field = None if self.bright_field_C.currentText() == 'n.a' else self.bright_field_C.currentIndex()
        Z_bright_field = self.bright_field_Z.currentIndex()
        red_channel = None if self.fluo_red.currentText() == 'n.a' else (self.fluo_red.currentIndex() - 1)
        green_channel = None if self.fluo_green.currentText() == 'n.a' else (self.fluo_green.currentIndex() - 1)
        blue_channel = None if self.fluo_blue.currentText() == 'n.a' else (self.fluo_blue.currentIndex() - 1)
        z_fluo = self.fluo_Z.currentIndex()
        if self.bright_field.isChecked():
            current_widget.setImageResource(image_resource_data,
                                            C=C_bright_field,
                                            Z=Z_bright_field,
                                            T=frame,
                                            )
            tab.setTabText(tab.currentIndex(), 'FOV bright field')
            if self.fluorescence.isChecked():
                current_widget.addLayer(name='fluorescence').setImage(image_resource_data,
                                                   C=(red_channel,
                                                      green_channel,
                                                      blue_channel),
                                                   Z=z_fluo,
                                                   T=frame,
                                                   alpha=True)
                tab.setTabText(tab.currentIndex(), 'FOV bf + fluo')
        elif self.fluorescence.isChecked():
            current_widget.setImageResource(image_resource_data,
                                            C=(red_channel,
                                               green_channel,
                                               blue_channel),
                                            Z=z_fluo,
                                            T=frame,
                                            )
            tab.setTabText(tab.currentIndex(), 'FOV fluorescence')
        else:
            current_widget.setImageResource(image_resource_data,
                                            C=C_bright_field,
                                            Z=Z_bright_field,
                                            T=frame,
                                            )
            tab.setTabText(tab.currentIndex(), 'FOV bright field')

        if last_widget is not None:
            current_widget.synchronize_with(last_widget)
        current_widget.draw_saved_rois(roi_list)
        PyDetecDiv.app.restoreOverrideCursor()


class DrawingToolsPalette(QDockWidget):
    """
    A dockable window with tools for drawing ROIs and other items.
    """

    def __init__(self, parent: MainWindow):
        super().__init__('Drawing tools', parent)
        self.setObjectName('Drawing_tools_palette')
        self.palette = QFrame()
        self.palette_layout = QVBoxLayout(self.palette)

        self.form = QFrame()
        self.form.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)

        self.formLayout = QGridLayout(self.form)
        self.formLayout.setObjectName("drawingToolsLayout")

        self.cursor_button = Cursor(self)
        self.draw_ROI_button = DrawRect(self)
        self.create_ROIs_button = DuplicateItem(self)
        self.draw_point = DrawPoint(self)
        self.tools = [self.cursor_button, self.draw_ROI_button, self.create_ROIs_button, self.draw_point]

        self.formLayout.addWidget(self.cursor_button, 0, 0)
        self.formLayout.addWidget(self.draw_ROI_button, 0, 1)
        self.formLayout.addWidget(self.create_ROIs_button, 0, 2)
        self.formLayout.addWidget(self.draw_point, 0, 3)

        self.form.setLayout(self.formLayout)

        self.properties = QFrame()
        self.properties.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)

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

        self.roi_width.valueChanged.connect(self.set_item_width)
        self.roi_height.valueChanged.connect(self.set_item_height)

        self.palette_layout.addWidget(self.form)
        self.palette_layout.addWidget(self.properties)
        self.setWidget(self.palette)

    def unset_tools(self) -> None:
        """
        Unset all available tools
        """
        for t in self.tools:
            t.setChecked(False)

    def current_tool(self) -> QToolButton | None:
        """
        Return the currently checked tool

        :return: the currently checked tool
        :rtype: QToolButton
        """
        for t in self.tools:
            if t.isChecked():
                return t
        return None

    def set_item_width(self, width: int):
        """
        Sets the width of the currently selected item, using the spinbox in drawing tools
        :param width: the desired width
        """
        PyDetecDiv.main_window.active_subwindow.currentWidget().scene.set_Item_width(width)

    def set_item_height(self, height: int):
        """
        Sets the height of the currently selected item, using the spinbox in drawing tools
        :param height: the desired height
        """
        PyDetecDiv.main_window.active_subwindow.currentWidget().scene.set_Item_height(height)


class Cursor(QToolButton):
    """
    QToolButton to activate the tool for selecting and dragging items in the view
    """

    def __init__(self, parent: DrawingToolsPalette):
        super().__init__(parent)
        self.parent = parent
        self.setIcon(QIcon(":icons/cursor"))
        self.setToolTip(DrawingTools.Cursor)
        self.setCheckable(True)
        self.clicked.connect(self.select_tool)
        self.setChecked(True)
        PyDetecDiv.current_drawing_tool = DrawingTools.Cursor

    def select_tool(self) -> None:
        """
        Select the Cursor tool
        """
        self.parent.unset_tools()
        self.setChecked(True)
        PyDetecDiv.current_drawing_tool = DrawingTools.Cursor


class DrawRect(QToolButton):
    """
    A QToolButton to activate the tool for drawing a ROI
    """

    def __init__(self, parent: DrawingToolsPalette):
        super().__init__(parent)
        self.parent = parent
        self.setIcon(QIcon(":icons/draw_Rect"))
        self.setToolTip(DrawingTools.DrawRect)
        self.setCheckable(True)
        self.clicked.connect(self.select_tool)

    def select_tool(self) -> None:
        """
        Select the DrawRect tool
        """
        self.parent.unset_tools()
        self.setChecked(True)
        PyDetecDiv.current_drawing_tool = DrawingTools.DrawRect


class DuplicateItem(QToolButton):
    """
    A QToolButton to activate the tool for duplicating a ROI
    """

    def __init__(self, parent: DrawingToolsPalette):
        super().__init__(parent)
        self.parent = parent
        self.setIcon(QIcon(":icons/duplicate_Item"))
        self.setToolTip(DrawingTools.DuplicateItem)
        self.setCheckable(True)
        self.clicked.connect(self.select_tool)

    def select_tool(self) -> None:
        """
        Select the DuplicateItem tool
        """
        self.parent.unset_tools()
        self.setChecked(True)
        PyDetecDiv.current_drawing_tool = DrawingTools.DuplicateItem

class DrawPoint(QToolButton):
    """
    A QToolButton to activate the tool for duplicating a ROI
    """

    def __init__(self, parent: DrawingToolsPalette):
        super().__init__(parent)
        self.parent = parent
        self.setIcon(QIcon(":icons/draw_Point"))
        self.setToolTip(DrawingTools.DrawPoint)
        self.setCheckable(True)
        self.clicked.connect(self.select_tool)

    def select_tool(self) -> None:
        """
        Select the DuplicateItem tool
        """
        self.parent.unset_tools()
        self.setChecked(True)
        PyDetecDiv.current_drawing_tool = DrawingTools.DrawPoint

class AnalysisToolsTree(QDockWidget):
    """
    A dockable window with tools for image analysis.
    """

    def __init__(self, parent: MainWindow):
        super().__init__('Analysis tools', parent)
        self.setObjectName('Analysis_tools_tree')
        tree_view = ToolboxTreeView()
        tree_view.setModel(ToolboxTreeModel(parent=self))
        self.setWidget(tree_view)
