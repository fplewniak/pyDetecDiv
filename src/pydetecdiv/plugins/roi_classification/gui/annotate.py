"""
ROI annotation for image classification
"""
import numpy as np
import pandas
import sqlalchemy
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QGraphicsTextItem, QDialogButtonBox, QFrame, QHBoxLayout, QLabel, QMenuBar
import pyqtgraph as pg

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.FOVmanager import FOVScene
from pydetecdiv.app.gui.core.widgets.viewers import Scene
from pydetecdiv.app.gui.core.widgets.viewers.images import ImageViewer
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer
from pydetecdiv.app.gui.core.widgets.viewers.plots import ChartView
from pydetecdiv.plugins.gui import ComboBox
from pydetecdiv.settings import get_config_value


def open_annotator(plugin, roi_selection, annotator):
    """
    Open an annotator instance with a selection of ROIs
    :param plugin: the plugin instance
    :param roi_selection: the list of ROIs to annotate
    """
    tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / ROI annotation')
    tab.project_name = PyDetecDiv.project_name
    # annotator = Annotator()
    annotator.setup(plugin=plugin)
    # tab.addTab(annotator, 'Annotation run')
    tab.set_top_tab(annotator, 'Annotation run')
    # tab.tabCloseRequested.connect(annotator.close)
    # plugin.gui.classes.setEnabled(False)
    # plugin.gui.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
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
        # self.setup()
        self.show_predictions = False

    @property
    def class_names(self):
        return self.plugin.class_names(as_string=False)

    def setup(self, menubar=None, plugin=None):
        super().setup(menubar=menubar)
        self.set_plugin(plugin)
        self.viewer_panel.setup(scene=AnnotatorScene())
        self.viewer_panel.setOrientation(Qt.Vertical)
        self.annotation_chart_view = AnnotationChartView(annotator=self)
        self.viewer_panel.addWidget(self.annotation_chart_view)
        self.zoom_set_value(100)
        self.video_frame.connect(self.plot_roi_classes)

        self.class_names_choice = []
        self.class_names_group = QActionGroup(self.menubar)
        self.class_names_group.setExclusive(True)

        for class_names in self.plugin.parameters.get('class_names').items:
            self.class_names_choice.append(QAction(class_names))
            self.class_names_choice[-1].setCheckable(True)
            if class_names == self.plugin.class_names():
                self.class_names_choice[-1].setChecked(True)
            self.class_names_group.addAction(self.class_names_choice[-1])
            self.menubar.menuClasses.addAction(self.class_names_choice[-1])
        self.class_names_group.triggered.connect(lambda x: self.update_ROI_selection(x.text()))

    # def closeEvent(self, event):
    #     self.plugin.gui.classes.setEnabled(True)
    #     self.plugin.gui.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

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
        self.tscale = roi_selection[0].fov.tscale * roi_selection[0].fov.tunit


    def next_roi(self):
        """
        Jumps to first frame of next ROI if there is one.
        """
        try:
            self.roi = next(self.roi_list)
            self.set_title(f'ROI: {self.roi.name}')
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
            # self.plugin.gui.classes.setEnabled(True)
            # self.plugin.gui.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
            pass

    def plot_roi_classes(self):
        self.annotation_chart_view.plot_roi_classes(self.roi_classes_idx)

    def update_ROI_selection(self, class_names):
        self.plugin.parameters.get('class_names').set_value(class_names)
        self.menubar.load_selected_ROIs()

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
        for frame, annotation in enumerate(self.plugin.get_prediction(self.roi, run=self.run, as_index=as_index)
                                           if self.show_predictions
                                           else self.plugin.get_annotation(self.roi, as_index=as_index)):
            if annotation != -1:
                roi_classes[frame] = annotation
        return roi_classes

    def annotate_current(self, class_name=None):
        """
        Assign the class name to the current frame
        :param class_name: the class name
        """
        self.roi_classes[self.T] = class_name
        self.roi_classes_idx[self.T] = self.class_names.index(class_name)
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
        # parameters.update(self.plugin.parameter_widgets.get_values('annotate'))
        parameters.update(self.plugin.parameters.values(groups='annotate'))
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
        if event.text() in list('azertyuiop')[0:len(self.class_names)]:
            self.annotate_current(self.class_names["azertyuiop".find(event.text())])
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
            # self.set_title(f'Annotation run {self.run.id_}')
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
        return self.annotator.class_names
        # return self.annotator.plugin.class_names(as_string=False)

    def plot_roi_classes(self, roi_classes_idx):
        self.chart().clear()
        self.chart().showAxes([True, True, True, True], [True, False, False, True])
        ticks = [(-1, 'n.a.')] + [(i, name) for i, name in enumerate(self.class_names)]
        left, right, bottom = self.chart().getAxis('left'), self.chart().getAxis('right'), self.chart().getAxis(
            'bottom')
        bottom.setLabel(units='frames')
        left.setTicks([ticks])
        self.chart().setLimits(xMin=0, xMax=len(roi_classes_idx), yMin=-1, yMax=len(self.class_names),
                               minYRange=len(self.class_names) + 1, maxYRange=len(self.class_names) + 1)
        right.setTicks([ticks])
        left.setGrid(100)
        self.addXline(self.annotator.T, angle=90, movable=False, pen=pg.mkPen('g', width=2))
        self.addLinePlot(roi_classes_idx, pen=pg.mkPen('k', width=1))
        self.addScatterPlot(roi_classes_idx, size=4, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))

    def clicked(self, plot, points):
        self.annotator.change_frame(int(points[0].pos().x()))


class AnnotationQualityCheck(Annotator):
    def setup(self, menubar=None, plugin=None):
        super().setup(menubar=menubar, plugin=plugin)
        self.control_panel.addWidget(AnnotationRunChooser(annotator=self))

class AnnotationRunChooser(QFrame):
    def __init__(self, parent=None, annotator=None):
        super().__init__(parent=parent)
        self.annotator = annotator
        self.setLayout(QHBoxLayout(self))
        self.layout().addWidget(QLabel('Run:'))
        all_runs = self.get_annotation_runs()
        all_runs.update(self.get_prediction_runs())
        self.layout().addWidget(ComboBox(self, items=all_runs,))


    def get_annotation_runs(self):
        """
       Get the class names for a project

       :return: the list of classes from the last annotation run for this project
       """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = list(project.repository.session.execute(
                sqlalchemy.text(f"SELECT "
                                f"run.parameters ->> '$.annotator' as annotator, "
                                f"run.parameters ->> '$.class_names' as class_names, "
                                f"run.id_ "
                                f"FROM run "
                                f"WHERE annotator='{get_config_value('project', 'user')}' "
                                f"AND (run.command='annotate_rois' OR run.command='import_annotated_rois') "
                                f"ORDER BY run.id_ DESC;")))
            # class_names = json.loads(results[-1][1])
            class_names_runs = {f'GT - {r[1]}': [rr[2] for rr in results if rr[1] == r[1]] for r in results}
        return class_names_runs

    def get_prediction_runs(self):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = list(project.repository.session.execute(
                sqlalchemy.text(f"SELECT "
                                f"run.parameters ->> '$.class_names' as class_names, "
                                f"run.id_ "
                                f"FROM run "
                                f"WHERE run.command='predict' "
                                f"ORDER BY run.id_ DESC;")))
            class_names_runs = {f'run {r[1]} {r[0]}': [] for r in results}
        return class_names_runs

class AnnotationMenuBar(QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.menuROI = self.addMenu('ROI selection')
        self.actionToggle_annotated = QAction('Annotated ROIs')
        self.actionToggle_annotated.setCheckable(True)
        self.actionToggle_annotated.changed.connect(self.load_selected_ROIs)
        self.menuROI.addAction(self.actionToggle_annotated)

        self.menuClasses = self.addMenu('ROI classes')

    def load_selected_ROIs(self):
        if self.actionToggle_annotated.isChecked():
            annotated_rois = self.parent().plugin.get_annotated_rois()
            self.parent().set_roi_list(annotated_rois)
        else:
            unannotated_rois, all_rois = self.parent().plugin.get_unannotated_rois()
            if unannotated_rois:
                self.parent().set_roi_list(unannotated_rois)
            else:
                self.parent().set_roi_list(all_rois)
        self.parent().next_roi()

class ClassificationViewer(Annotator):
    def __init__(self):
        super().__init__()

class ClassificationMenuBar(AnnotationMenuBar):
    def __init__(self, parent):
        super().__init__(parent)

    def load_selected_ROIs(self):
        runs = self.parent().plugin.get_prediction_runs()
        classified_rois = self.parent().plugin.get_annotated_rois(runs[self.parent().plugin.class_names()][-1])
        self.parent().set_roi_list(classified_rois)
        self.parent().next_roi()
        # if self.actionToggle_annotated.isChecked():
        #     annotated_rois = self.parent().plugin.get_annotated_rois()
        #     self.parent().set_roi_list(annotated_rois)
        # else:
        #     unannotated_rois, all_rois = self.parent().plugin.get_unannotated_rois()
        #     if unannotated_rois:
        #         self.parent().set_roi_list(unannotated_rois)
        #     else:
        #         self.parent().set_roi_list(all_rois)
        # self.parent().next_roi()
