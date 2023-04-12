"""
Classes for persistent windows of the GUI
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QMainWindow, QMdiArea, QTabWidget, QDockWidget, QFormLayout, QLabel, QComboBox, \
    QDialogButtonBox, QWidget, QFrame
from pydetecdiv.app.gui import MainToolBar, MainStatusBar, FileMenu, DataMenu
from pydetecdiv.app import get_settings, PyDetecDiv, pydetecdiv_project

from pydetecdiv.app.gui.ImageViewer import ImageViewer

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
        # self.mdi_area.addSubWindow(self.viewer)
        # self.viewer.showMaximized()

        self.image_resource_selector = ImageResourceChooser(self, )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.image_resource_selector, Qt.Vertical)

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

    def add_tabbbed_viewer(self, title):
        if title not in self.tabs:
            self.tabs[title] = TabbedViewer(title)
            self.mdi_area.addSubWindow(self.tabs[title])
            self.tabs[title].show()
        return self.tabs[title]

class TabbedViewer(QTabWidget):
    def __init__(self, title):
        super().__init__()
        self.viewer = ImageViewer()
        self.setWindowTitle(title)
        self.setDocumentMode(True)
        self.addTab(self.viewer, 'Image viewer')

class ImageResourceChooser(QDockWidget):
    def __init__(self, parent):
        super().__init__('Image resource selector', parent)
        self.setObjectName('Image_resource_selector')
        self.form = QFrame()

        self.formLayout = QFormLayout(self.form)
        self.formLayout.setObjectName(u"formLayout")
        self.position_label = QLabel('Position', self.form)
        self.position_label.setObjectName(u"position_label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.position_label)

        self.position_choice = QComboBox(self.form)
        self.position_choice.setObjectName(u"position_choice")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.position_choice)

        self.stage_label = QLabel('Stage', self.form)
        self.stage_label.setObjectName(u"stage_label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.stage_label)

        self.stage_choice = QComboBox(self.form)
        self.stage_choice.setObjectName(u"stage_choice")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.stage_choice)

        self.OK_button = QDialogButtonBox(self.form)
        self.OK_button.setObjectName(u"OK_button")
        self.OK_button.setStandardButtons(QDialogButtonBox.Ok)

        self.formLayout.setWidget(2, QFormLayout.SpanningRole, self.OK_button)

        self.setWidget(self.form)

        PyDetecDiv().project_selected.connect(self.set_choice)
        self.OK_button.accepted.connect(self.accept)

    def set_choice(self, p_name):
        with pydetecdiv_project(p_name) as project:
            FOV_list = [fov.name for fov in project.get_objects('FOV')]
            dataset_list = [ds.name for ds in project.get_objects('Dataset')]
            self.position_choice.clear()
            self.position_choice.addItems(sorted(FOV_list))
            self.stage_choice.clear()
            self.stage_choice.addItems(dataset_list)


    def accept(self):
        PyDetecDiv().setOverrideCursor(QCursor(Qt.WaitCursor))
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            fov = project.get_named_object('FOV', self.position_choice.currentText())
            dataset = project.get_named_object('Dataset', self.stage_choice.currentText()).name
            image_resource = fov.image_resource(dataset)
        tab_key = f'{PyDetecDiv().project_name}/{fov.name}/{dataset}'
        tab = self.parent().add_tabbbed_viewer(tab_key)
        tab.setWindowTitle(tab_key)
        tab.viewer.set_image_resource(image_resource)
        PyDetecDiv().setOverrideCursor(QCursor(Qt.ArrowCursor))
