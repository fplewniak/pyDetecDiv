from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenuBar, QGraphicsTextItem
import pyqtgraph as pg

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers import Scene
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer
from pydetecdiv.app.gui.core.widgets.viewers.plots import ChartView
from pydetecdiv.settings import get_config_value


class AnnotationTool(VideoPlayer):
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
        self.roi_classes_idx = []
        self.class_item = None
        self.annotation_chart_view = None
        self.show_predictions = False

    @property
    def class_names(self):
        return self.plugin.class_names(as_string=False)

    @property
    def annotation_run_list(self):
        raise NotImplementedError

    def roi_classes(self, frame):
        if self.roi_classes_idx[frame] == -1:
            return '-'
        return self.class_names[self.roi_classes_idx[frame]]

    def setup(self, menubar=None, plugin=None, scene=None):
        super().setup(menubar=menubar)
        self.plugin = plugin
        self.viewer_panel.setup(scene=scene)
        self.viewer_panel.setOrientation(Qt.Vertical)
        self.annotation_chart_view = AnnotationChartView(annotator=self)
        self.viewer_panel.addWidget(self.annotation_chart_view)
        self.zoom_set_value(100)
        self.video_frame.connect(self.plot_roi_classes)

    def set_title(self, title):
        self.parent().parent().setTabText(self.parent().parent().currentIndex(), title)

    def set_run_list(self):
        raise NotImplementedError

    def change_frame(self, T=0):
        """
        Change frame and display the corresponding class name below the image
        :param T: the frame index
        """
        super().change_frame(T)
        self.display_class_name()

    def display_class_name(self, roi_class=None):
        """
        Display the class name below the frame
        :param roi_class: the class name
        """
        if self.class_item is not None:
            self.scene.removeItem(self.class_item)
        self.class_item = self.viewer.background.addItem(QGraphicsTextItem('-'))

        roi_class = self.roi_classes(self.T) if roi_class is None else roi_class
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

    def plot_roi_classes(self):
        self.annotation_chart_view.plot_roi_classes(self.roi_classes_idx)

    def update_ROI_selection(self, class_names):
        self.plugin.parameters.get('class_names').set_value(class_names)
        self.load_selected_ROIs()

    def load_selected_ROIs(self):
        raise NotImplementedError

    def update_roi_classes_plot(self):
        self.annotation_chart_view.chart().clear()
        self.plot_roi_classes()

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
                self.roi_classes_idx = self.get_roi_annotations(as_index=True)
                self.plot_roi_classes()
                self.change_frame(0)
                self.video_frame.emit(0)
                self.control_panel.video_control.t_slider.setSliderPosition(0)
                PyDetecDiv.main_window.active_subwindow.setCurrentWidget(self)
        except StopIteration:
            pass

    def get_roi_annotations(self, as_index=False):
        """
        Retrieve from the database the manual annotations for a ROI
        """
        if as_index:
            roi_classes = [-1] * self.viewer.image_resource_data.sizeT
        else:
            roi_classes = ['-'] * self.viewer.image_resource_data.sizeT
        for frame, annotation in enumerate(
                self.plugin.get_classifications(self.roi, self.annotation_run_list, as_index=as_index)):
            if annotation != -1:
                roi_classes[frame] = annotation
        return roi_classes


class AnnotationScene(Scene):
    """
    The viewer scene where images and other items are drawn
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def annotation_tool(self):
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


class ManualAnnotator(AnnotationTool):
    def __init__(self):
        super().__init__()

    @property
    def annotation_run_list(self):
        annotation_runs = self.plugin.get_annotation_runs()
        return annotation_runs[self.plugin.class_names()]

    def setup(self, menubar=None, plugin=None, scene=None):
        if scene is None:
            scene = AnnotationScene()
        super().setup(menubar=ManualAnnotationMenuBar(self), plugin=plugin, scene=scene)
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
        elif event.key() == Qt.Key_PageDown:
            self.annotate_current(class_name=f'{self.class_item.toPlainText()}')
            self.class_item.setDefaultTextColor('black')
            if self.run is None:
                self.save_run()
            self.plugin.save_annotations(self.roi, [self.class_names[c] for c in self.roi_classes_idx], self.run)
            self.next_roi()
        elif event.key() == Qt.Key_Escape:
            self.next_roi()

    def load_selected_ROIs(self):
        if self.menubar.actionToggle_annotated.isChecked():
            annotated_rois = self.plugin.get_annotated_rois()
            self.set_roi_list(annotated_rois)
        else:
            unannotated_rois, all_rois = self.plugin.get_unannotated_rois()
            if unannotated_rois:
                self.set_roi_list(unannotated_rois)
            else:
                self.menubar.actionToggle_annotated.setChecked(True)
                self.set_roi_list(all_rois)
        self.next_roi()

    def annotate_current(self, class_name=None):
        """
        Assign the class name to the current frame
        :param class_name: the class name
        """
        self.roi_classes_idx[self.T] = self.class_names.index(class_name)
        self.update_roi_classes_plot()

    def save_run(self):
        """
        Save the current ROI annotation process in the database
        """
        parameters = {'annotator': get_config_value('project', 'user'), }
        parameters.update(self.plugin.parameters.values(groups='annotate'))
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.run = self.plugin.save_run(project, 'annotate_rois', parameters)


class ManualAnnotationMenuBar(QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.menuROI = self.addMenu('ROI selection')
        self.actionToggle_annotated = QAction('Annotated ROIs')
        self.actionToggle_annotated.setCheckable(True)
        self.actionToggle_annotated.changed.connect(self.parent().load_selected_ROIs)
        self.menuROI.addAction(self.actionToggle_annotated)

        self.menuClasses = self.addMenu('ROI classes')


class ClassificationViewer(AnnotationTool):
    def __init__(self):
        super().__init__()


class ClassificationMenuBar(QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
