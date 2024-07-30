"""
ROI annotation for image classification
"""
import numpy as np
import pandas
from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import QGraphicsTextItem, QDialogButtonBox
import pyqtgraph as pg

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.FOVmanager import FOVScene
from pydetecdiv.app.gui.core.widgets.viewers import Scene
from pydetecdiv.app.gui.core.widgets.viewers.images import ImageViewer
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer
from pydetecdiv.app.gui.core.widgets.viewers.plots import ChartView
from pydetecdiv.settings import get_config_value


def open_annotator(plugin, roi_selection):
    """
    Open an annotator instance with a selection of ROIs
    :param plugin: the plugin instance
    :param roi_selection: the list of ROIs to annotate
    """
    tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / ROI annotation')
    tab.project_name = PyDetecDiv.project_name
    annotator = Annotator()
    annotator.set_plugin(plugin)
    # tab.addTab(annotator, 'Annotation run')
    tab.set_top_tab(annotator, 'Annotation run')
    # tab.tabCloseRequested.connect(annotator.close)
    plugin.gui.classes.setEnabled(False)
    plugin.gui.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
    annotator.set_roi_list(roi_selection)
    annotator.tscale = roi_selection[0].fov.tscale * roi_selection[0].fov.tunit
    annotator.next_roi()
    annotator.setFocus()


class Annotator(VideoPlayer):
    """
    Annotator class extending the VideoPlayer class to define functionalities specific to ROI image annotation
    """

    def __init__(self):
        super().__init__()
        self.roi_list = None
        self.roi = None
        self.run = None
        self.plugin = None
        self.viewport_rect = None
        self.roi_classes = []
        self.class_item = None  # QGraphicsTextItem('-')
        self.annotation_chart_view = None
        self.setup()
        self.video_frame.connect(self.plot_roi_classes)

    def setup(self, menubar=None):
        super().setup(menubar=menubar)
        self.viewer_panel.setup(scene=AnnotatorScene())
        self.viewer_panel.setOrientation(Qt.Vertical)
        self.annotation_chart_view = AnnotationChartView(annotator=self)
        self.viewer_panel.addWidget(self.annotation_chart_view)
        self.zoom_set_value(200)

    def closeEvent(self, event):
        self.plugin.gui.classes.setEnabled(True)
        self.plugin.gui.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def set_plugin(self, plugin):
        """
        Define the plugin instance to enable the annotator to access some data
        :param plugin: the plugin instance
        """
        self.plugin = plugin

    def set_title(self, title):
        self.parent().parent().setTabText(self.parent().parent().currentIndex(), title)

    def set_roi_list(self, roi_selection):
        """
        Sets the list of ROIs to annotate as an iterator
        :param roi_selection: the list of ROIs
        """
        self.roi_list = iter(roi_selection)

    def next_roi(self):
        """
        Jumps to first frame of next ROI if there is one.
        """
        try:
            self.roi = next(self.roi_list)
            with pydetecdiv_project(PyDetecDiv.project_name) as project:
                image_resource = project.get_linked_objects('FOV', self.roi)[0].image_resource()
                x1, x2 = self.roi.top_left[0], self.roi.bottom_right[0] + 1
                y1, y2 = self.roi.top_left[1], self.roi.bottom_right[1] + 1
                crop = (slice(x1, x2), slice(y1, y2))
                self.setBackgroundImage(image_resource.image_resource_data(), crop=crop)
                self.viewer.display()
                self.roi_classes = self.get_roi_annotations()
                self.roi_classes_idx = self.get_roi_annotations(as_index=True)
                self.plot_roi_classes()
                self.change_frame(0)
                self.video_frame.emit(0)
                self.control_panel.video_control.t_slider.setSliderPosition(0)
                # self.ui.view_name.setText(f'ROI: {self.roi.name}')
                PyDetecDiv.main_window.active_subwindow.setCurrentWidget(self)
        except StopIteration:
            self.plugin.gui.classes.setEnabled(True)
            self.plugin.gui.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
            pass

    def plot_roi_classes(self):
        self.annotation_chart_view.plot_roi_classes(self.roi_classes_idx)

    def update_roi_classes_plot(self):
        self.annotation_chart_view.chart().clear()
        self.plot_roi_classes()

    def get_roi_annotations(self, as_index=False):
        """
        Retrieve from the database the manual annotations for a ROI
        """
        if as_index:
            roi_classes = [-1] * self.viewer.image_resource_data.sizeT
        else:
            roi_classes = ['-'] * self.viewer.image_resource_data.sizeT
        for frame, annotation in enumerate(self.plugin.get_annotation(self.roi, as_index=as_index)):
            if annotation != -1:
                roi_classes[frame] = annotation
        return roi_classes

    def annotate_current(self, class_name=None):
        """
        Assign the class name to the current frame
        :param class_name: the class name
        """
        self.roi_classes[self.T] = class_name
        self.roi_classes_idx[self.T] = self.plugin.class_names.index(class_name)
        self.update_roi_classes_plot()

    def display_class_name(self, roi_class=None):
        """
        Display the class name below the frame
        :param roi_class: the class name
        """
        if self.class_item is not None:
            self.scene.removeItem(self.class_item)
        self.class_item = self.viewer.background.addItem(QGraphicsTextItem('-'))

        if self.roi_classes[self.T] == '-' and self.T > 0 and self.roi_classes[self.T - 1] != '-':
            roi_class = self.roi_classes[self.T - 1]
            self.class_item.setDefaultTextColor('red')
        else:
            roi_class = self.roi_classes[self.T] if roi_class is None else roi_class
            self.class_item.setDefaultTextColor('black')
        self.class_item.setPlainText(roi_class)
        text_boundingRect = self.class_item.boundingRect()
        frame_boundingRect = self.viewer.background.image.boundingRect()
        self.class_item.setPos(frame_boundingRect.x() + (frame_boundingRect.width() - text_boundingRect.width()) / 2,
                               frame_boundingRect.y() + frame_boundingRect.height())
        self.viewport_rect = QRectF(min([self.class_item.x(), self.viewer.background.image.x()]),
                                    min([self.class_item.y(), self.viewer.background.image.y()]) - 5,
                                    max([text_boundingRect.width(), frame_boundingRect.width()]),
                                    text_boundingRect.height() + frame_boundingRect.height() + 5,
                                    )

    def change_frame(self, T=0):
        """
        Change frame and display the corresponding class name below the image
        :param T: the frame index
        """
        super().change_frame(T)
        self.display_class_name()

    def save_run(self):
        """
        Save the current ROI annotation process in the database
        """
        parameters = {'annotator': get_config_value('project', 'user'), }
        parameters.update(self.plugin.parameters.get_values('annotate'))
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.run = self.plugin.save_run(project, 'annotate_rois', parameters)

    # def focusInEvent(self, event):
    #     """
    #     When the scene is in focus, then draw a larger frame around it to indicate its in-focus status
    #     :param event: the focusInEvent
    #     """
    #     print('Focus in Annotator')
    #
    # def focusOutEvent(self, event):
    #     """
    #     When the scene is out of focus, then draw a small frame around it to indicate its out-focus status
    #     :param event: the focusOutEvent
    #     """
    #     print('Focus out Annotator')

    def keyPressEvent(self, event):
        """
                Handle actions triggered by pressing keys when the scene is in focus.
                Letters from azertyuiop assign a class to the current frame, and jumps to the next frame suggesting a class
                Space bar validates the suggested assignation and jumps to the next frame
                Right arrow moves one frame forward
                Left arrow moves one frame backwards
                Enter key validates the current suggestion, saves the annotations to the database and jumps to the nex ROI if
                there is one
                Escape key cancels annotations and jumps to the next ROI if there is one
                :param event: the keyPressEvent
                """
        if event.text() in list('azertyuiop')[0:len(self.plugin.class_names)]:
            self.annotate_current(self.plugin.class_names["azertyuiop".find(event.text())])
            self.change_frame(min(self.T + 1, self.viewer.image_resource_data.sizeT - 1))
        elif event.text() == ' ':
            self.annotate_current(class_name=f'{self.class_item.toPlainText()}')
            self.change_frame(min(self.T + 1, self.viewer.image_resource_data.sizeT - 1))
        elif event.key() == Qt.Key_Right:
            self.change_frame(min(self.T + 1, self.viewer.image_resource_data.sizeT - 1))
        elif event.key() == Qt.Key_Left:
            self.change_frame(max(self.T - 1, 0))
        # elif event.key() == Qt.Key_Enter:
        #     print('Enter')
        # elif event.key() == Qt.Key_Return:
        #     print('Return')
        elif event.key() == Qt.Key_PageDown:
            self.annotate_current(class_name=f'{self.class_item.toPlainText()}')
            self.class_item.setDefaultTextColor('black')
            if self.run is None:
                self.save_run()
            self.set_title(f'Annotation run {self.run.id_}')
            self.plugin.save_annotations(self.roi, self.roi_classes, self.run)
            self.next_roi()
        elif event.key() == Qt.Key_Escape:
            self.next_roi()


class AnnotatorScene(Scene):
    """
    The viewer scene where images and other items are drawn
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def annotator(self):
        return self.viewer.parent().parent()

    def mouseMoveEvent(self, event):
        pass

    def mousePressEvent(self, event):
        pass


class AnnotationChartView(ChartView):
    def __init__(self, parent=None, annotator=None):
        super().__init__(parent=parent)
        self.annotator = annotator

    @property
    def class_names(self):
        return self.annotator.plugin.class_names

    def plot_roi_classes(self, roi_classes_idx):
        self.chart().clear()
        self.chart().showAxes([True, True, True, True], [True, False, False, True])
        ticks = [(-1, 'n.a.')] + [(i, name) for i, name in enumerate(self.class_names)]
        left, right, bottom = self.chart().getAxis('left'), self.chart().getAxis('right'), self.chart().getAxis('bottom')
        bottom.setLabel(units='frames')
        left.setTicks([ticks])
        self.chart().setLimits(xMin=0, xMax=len(roi_classes_idx), yMin=-1, yMax=len(self.class_names),
                          minYRange=len(self.class_names)+1, maxYRange=len(self.class_names)+1)
        right.setTicks([ticks])
        left.setGrid(100)
        self.addXline(self.annotator.T, angle=90, movable=False, pen=pg.mkPen('g', width=2))
        self.addLinePlot(roi_classes_idx, pen=pg.mkPen('k', width=1))
        self.addScatterPlot(roi_classes_idx, size=4, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))

    def clicked(self, plot, points):
        self.annotator.change_frame(int(points[0].pos().x()))
