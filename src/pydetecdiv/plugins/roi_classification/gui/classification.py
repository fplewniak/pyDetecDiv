"""
GUI for ROI manual annotation and class visualization
"""
import json
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QRectF, QItemSelectionModel
from PySide6.QtGui import QAction, QActionGroup, QContextMenuEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QMenuBar, QGraphicsTextItem, QPushButton, QDialogButtonBox, QMenu, QFileDialog
import pyqtgraph as pg

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers import Scene
from pydetecdiv.app.gui.core.widgets.viewers.images.video import VideoPlayer
from pydetecdiv.app.gui.core.widgets.viewers.plots import ChartView
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.plugins.gui import ListView, Dialog
from pydetecdiv.settings import get_config_value
from pydetecdiv.utils import BidirectionalIterator, previous
from pydetecdiv.plugins import Plugin
from pydetecdiv.app.parameters import Parameter
from pydetecdiv.plugins.roi_classification.utils import get_classifications, get_annotation_runs



class AnnotationScene(Scene):
    """
    The viewer scene where images and other items are drawn
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def annotation_tool(self):
        """
        Returns the annotation tool parent of this annotation scene
        :return: the annotation tool
        """
        return self.viewer.parent().parent()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pass

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pass


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
    def class_names(self) -> list[str]:
        """
        The current list of class names

        :return: The current list of class names
        """
        return self.plugin.class_names(as_string=False)

    @property
    def annotation_run_list(self):
        """
        The list of annotation and/or prediction run ids available in the project
        """
        raise NotImplementedError

    def roi_class(self, frame: int) -> str:
        """
        The class name of the current ROI at the given frame

        :param frame: the frame index
        :return: the class name
        """
        if self.roi_classes_idx[frame] == -1:
            return '-'
        return self.class_names[self.roi_classes_idx[frame]]

    def setup(self, menubar: QMenuBar = None, plugin: Plugin = None, scene: AnnotationScene = None) -> None:
        """
        Sets the Annotation tool up

        :param menubar: the menu bar
        :param plugin: the plugin
        :param scene: the annotation scene to display plots in
        """
        super().setup(menubar=menubar)
        if plugin is not None:
            self.plugin = plugin
        self.menubar.setup()
        self.viewer_panel.setup(scene=scene)
        self.viewer_panel.setOrientation(Qt.Orientation.Vertical)
        self.annotation_chart_view = AnnotationChartView(annotator=self)
        self.viewer_panel.addWidget(self.annotation_chart_view)
        self.zoom_set_value(100)
        self.video_frame.connect(self.plot_roi_classes)

    def set_title(self, title: str) -> None:
        """
        Sets the title of the Annotation tool

        :param title: the title to display
        """
        self.parent().parent().setTabText(self.parent().parent().currentIndex(), title)

    def set_run_list(self) -> None:
        """
        Abstract method for setting the list of annotation or prediction runs that use the specified class names
        """
        raise NotImplementedError

    def change_frame(self, T: int = 0) -> None:
        """
        Change frame and display the corresponding class name below the image
        :param T: the frame index
        """
        super().change_frame(T)
        self.display_class_name()

    def display_class_name(self, roi_class: str = None) -> None:
        """
        Display the class name below the frame
        :param roi_class: the class name
        """
        if self.class_item is not None:
            self.scene.removeItem(self.class_item)
        self.class_item = self.viewer.background.addItem(QGraphicsTextItem('-'))

        roi_class = self.roi_class(self.T) if roi_class is None else roi_class
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

    def plot_roi_classes(self) -> None:
        """
        A convenience method to plot the class annotations or predictions for a list of ROIs
        """
        self.annotation_chart_view.plot_roi_classes(self.roi_classes_idx)

    def select_class_names(self, class_names: str) -> None:
        """
        Select the set of classes

        :param class_names: the set of classes
        """
        self.plugin.parameters['class_names'].set_value(class_names)

    def update_ROI_selection(self, class_names: str) -> None:
        """
        Updates the ROI selection according to a set of classes.

        :param class_names: the set of classes
        """
        self.select_class_names(class_names)
        self.load_selected_ROIs()

    def load_selected_ROIs(self) -> None:
        """
        Abstract method loading selected ROIs. This method is executed when the set of classes has been changed to select the ROIs
        that have been annotated or classified using those classes
        """
        raise NotImplementedError

    def update_roi_classes_plot(self) -> None:
        """
        Updates the chart view plot whenever there has been a modification
        """
        self.annotation_chart_view.chart().clear()
        self.plot_roi_classes()

    def set_roi_list(self, roi_selection: list[ROI]) -> None:
        """
        Sets the list of ROIs to annotate as an iterator
        :param roi_selection: the list of ROIs
        """
        # self.roi_list = iter(roi_selection)
        self.roi_list = BidirectionalIterator(roi_selection)
        self.tscale = roi_selection[0].fov.tscale * roi_selection[0].fov.tunit

    def next_roi(self) -> None:
        """
        Jumps to first frame of next ROI if there is one.
        """
        try:
            self.roi = next(self.roi_list)
            self.display_roi()
        except StopIteration:
            pass

    def previous_roi(self) -> None:
        """
        Jumps back to the first frame of previous ROI in the list
        """
        try:
            self.roi = previous(self.roi_list)
            self.display_roi()
        except StopIteration:
            pass

    def display_roi(self) -> None:
        """
        Displays a ROI image
        """
        self.set_title(f'ROI: {self.roi.name}')
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            image_resource = project.get_linked_objects('FOV', self.roi)[0].image_resource()
            x1, x2 = self.roi.top_left[0], self.roi.bottom_right[0] + 1
            y1, y2 = self.roi.top_left[1], self.roi.bottom_right[1] + 1
            crop = (slice(x1, x2), slice(y1, y2))
            self.setBackgroundImage(image_resource.image_resource_data(), crop=crop)
            self.viewer.display()
            self.roi_classes_idx = self.get_roi_annotations()
            self.plot_roi_classes()
            self.change_frame(0)
            self.video_frame.emit(0)
            self.control_panel.video_control.t_slider.setSliderPosition(0)
            PyDetecDiv.main_window.active_subwindow.setCurrentWidget(self)

    def get_roi_annotations(self) -> list[int]:
        """
        Retrieve from the database the manual annotations for a ROI
        """
        roi_classes = [-1] * self.viewer.image_resource_data.sizeT
        for frame, annotation in enumerate(get_classifications(self.roi, self.annotation_run_list, as_index=True)):
            if annotation != -1:
                roi_classes[frame] = annotation
        return roi_classes


class AnnotationChartView(ChartView):
    """
    Generic Annotation viewer providing features common to chart views in Annotator and Prediction viewer
    """

    def __init__(self, parent=None, annotator=None):
        super().__init__(parent=parent)
        self.annotator = annotator

    @property
    def class_names(self) -> list[str]:
        """
        The list of class names

        :return: The list of class names
        """
        return self.annotator.class_names

    def plot_roi_classes(self, roi_classes_idx: list[int]) -> None:
        """
        Plots ROI classes along time for a selection of ROIs, one ROI after the other

        :param roi_classes_idx: the list of ROI ids to display classes
        """
        self.chart().clear()
        self.chart().showAxes((True, True, True, True), [True, False, False, True])
        # ticks = [(-1, 'n.a.')] + [(i, name) for i, name in enumerate(self.class_names)]
        ticks = [(-1, 'n.a.')] + list(enumerate(self.class_names))
        left, right, bottom = self.chart().getAxis('left'), self.chart().getAxis('right'), self.chart().getAxis(
                'bottom')
        bottom.setLabel(units='frames')
        left.setTicks([ticks])
        self.chart().setLimits(xMin=0, xMax=len(roi_classes_idx), yMin=-1, yMax=len(self.class_names),
                               minYRange=len(self.class_names) + 1, maxYRange=len(self.class_names) + 1)
        right.setTicks([ticks])
        left.setGrid(100)
        self.addXline(self.annotator.T, angle=90, movable=False, pen=pg.mkPen('g', width=2))
        self.addLinePlot(np.ndarray(roi_classes_idx), pen=pg.mkPen('k', width=1))
        self.addScatterPlot(np.ndarray(roi_classes_idx), size=6, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))

    def clicked(self, plot: pg.ScatterPlotItem, points: list[pg.SpotItem]) -> None:
        """
        If a point representing the annotation of the ROI for a given frame has been clicked, then the time cursor is moved to the
        clicked frame

        :param plot: the ScatterPlotItem displaying the annotation plot
        :param points: the clicked point
        """
        # print(points)
        self.annotator.change_frame(int(points[0].pos().x()))


class ManualAnnotator(AnnotationTool):
    """
    Class to annotate ROIs along time
    """

    def __init__(self):
        super().__init__()
        self.define_classes_dialog = None

    @property
    def annotation_run_list(self) -> list[int]:
        """
        The list of annotation runs available in the project (manual annotation and annotation import)

        :return: the list of Run ids
        """
        annotation_runs = get_annotation_runs()
        return annotation_runs[self.plugin.parameters['class_names'].key]

    def setup(self, menubar: QMenuBar = None, plugin: Plugin = None, scene: AnnotationScene = None) -> None:
        """
        Sets the Manual annotator up

        :param menubar: the menu bar
        :param plugin: the plugin
        :param scene: the annotation scene to display plots in
        """
        if scene is None:
            scene = AnnotationScene()
        super().setup(menubar=ManualAnnotationMenuBar(self), plugin=plugin, scene=scene)

    def keyPressEvent(self, event: QKeyEvent) -> None:
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
        elif event.key() == Qt.Key.Key_Right:
            self.change_frame(min(self.T + 1, self.viewer.image_resource_data.sizeT - 1))
        elif event.key() == Qt.Key.Key_Left:
            self.change_frame(max(self.T - 1, 0))
        elif event.key() == Qt.Key.Key_PageDown:
            self.annotate_current(class_name=f'{self.class_item.toPlainText()}')
            # self.class_item.setDefaultTextColor('black')
            if self.run is None:
                self.save_run()
            # self.plugin.save_annotations(self.roi, [self.class_names[c] for c in self.roi_classes_idx], self.run)
            self.plugin.save_annotations(self.roi, self.roi_classes_idx, self.run)
            self.next_roi()
        elif event.key() == Qt.Key.Key_PageUp:
            self.previous_roi()
        elif event.key() == Qt.Key.Key_Escape:
            self.next_roi()

    def define_classes(self, suggestion: list[str] = None) -> None:
        """
        Launch the class set definition interface

        :param suggestion: the default list of classes
        """
        if self.define_classes_dialog is None:
            self.define_classes_dialog = DefineClassesDialog(self, self.plugin, suggestion=suggestion)
        else:
            self.define_classes_dialog.setup_class_names(suggestion=suggestion)
            self.define_classes_dialog.show()

    def load_selected_ROIs(self) -> None:
        """
        Loads the annotated and/or annotated ROIs and shows the first ROI classes accordingly
        """
        if self.menubar.actionToggle_annotated.isChecked():
            annotated_rois = self.plugin.get_annotated_rois()
            if annotated_rois:
                self.set_roi_list(annotated_rois)
            else:
                self.menubar.actionToggle_annotated.setChecked(False)
                unannotated_rois, all_rois = self.plugin.get_unannotated_rois()
                self.set_roi_list(all_rois)
        else:
            unannotated_rois, all_rois = self.plugin.get_unannotated_rois()
            if unannotated_rois:
                self.set_roi_list(unannotated_rois)
            else:
                self.menubar.actionToggle_annotated.setChecked(True)
                self.set_roi_list(all_rois)
        self.next_roi()

    def annotate_current(self, class_name: str = None) -> None:
        """
        Assign the class name to the current frame
        :param class_name: the class name
        """
        if class_name in self.class_names:
            self.roi_classes_idx[self.T] = self.class_names.index(class_name)
            self.update_roi_classes_plot()

    def save_run(self) -> None:
        """
        Save the current ROI annotation process in the database
        """
        parameters = {'annotator': get_config_value('project', 'user'), }
        parameters.update(self.plugin.parameters.values(groups='annotate'))
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.run = self.plugin.save_run(project, 'annotate_rois', parameters)


class PredictionViewer(AnnotationTool):
    """
    Class to visualize predictions for ROIs along time
    """

    def setup(self, menubar: QMenuBar = None, plugin: Plugin = None, scene: AnnotationScene = None) -> None:
        """
        Sets the Viewer up

        :param menubar: the menu bar
        :param plugin: the plugin
        :param scene: the annotation scene to display plots in
        """
        if scene is None:
            scene = AnnotationScene()
        super().setup(menubar=PredictionMenuBar(self), plugin=plugin, scene=scene)

    @property
    def annotation_run_list(self) -> list[int]:
        """
        Gets the list of available annotation runs

        :return: the list of annotation runs
        """
        return [self.menubar.prediction_runs_group.checkedAction().data()]

    def load_selected_ROIs(self) -> None:
        """
        Selects the prediction run checked in the menu
        """
        self.select_prediction_run(self.menubar.prediction_runs_group.checkedAction())

    def select_prediction_run(self, prediction_run: QAction) -> None:
        """
        Loads the ROIs corresponding to the selected prediction run and shows ROI class predictions accordingly

        :param prediction_run: the requested QAction of prediction runs menu
        """
        classified_rois = self.plugin.get_annotated_rois(run=prediction_run.data())
        self.set_roi_list(classified_rois)
        self.next_roi()

    def keyPressEvent(self, event: QKeyEvent) -> None:
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
        if event.text() == ' ':
            self.change_frame(min(self.T + 1, self.viewer.image_resource_data.sizeT - 1))
        elif event.key() == Qt.Key.Key_Right:
            self.change_frame(min(self.T + 1, self.viewer.image_resource_data.sizeT - 1))
        elif event.key() == Qt.Key.Key_Left:
            self.change_frame(max(self.T - 1, 0))
        elif event.key() == Qt.Key.Key_PageDown:
            self.next_roi()
        elif event.key() == Qt.Key.Key_PageUp:
            self.previous_roi()
        elif event.key() == Qt.Key.Key_Escape:
            self.next_roi()


class AnnotationMenuBar(QMenuBar):
    """
    The menu bar for annotation tools
    """

    def __init__(self, parent: AnnotationTool):
        super().__init__(parent)
        self.class_names_choice = []
        self.menuClasses = self.addMenu('ROI classes')
        self.class_names_group = QActionGroup(self)
        self.class_names_group.setExclusive(True)
        self.class_names_group.triggered.connect(lambda x: self.parent().update_ROI_selection(x.text()))

    def setup(self) -> None:
        """
        Sets the menus up
        """
        self.set_class_names_choice()

    def set_class_names_choice(self) -> None:
        """
        Sets the options for class names, i.e. all lists of class names that are available in the current project.
        """
        for class_names in self.parent().plugin.parameters['class_names'].keys:
            self.class_names_choice.append(QAction(class_names))
            self.class_names_choice[-1].setCheckable(True)
            if class_names == self.parent().plugin.class_names():
                self.class_names_choice[-1].setChecked(True)
            self.class_names_group.addAction(self.class_names_choice[-1])
            self.menuClasses.addAction(self.class_names_choice[-1])


class ManualAnnotationMenuBar(AnnotationMenuBar):
    """
    The menu bar for a manual annotator widget
    """

    def __init__(self, parent: ManualAnnotator):
        super().__init__(parent)
        self.menuROI = self.addMenu('ROI selection')
        self.actionToggle_annotated = QAction('Annotated ROIs')
        self.actionToggle_annotated.setCheckable(True)
        self.actionToggle_annotated.changed.connect(self.parent().load_selected_ROIs)
        self.action_edit_classes = QAction('Edit class sets')
        self.menuROI.addAction(self.actionToggle_annotated)
        # self.menuClasses.addSeparator()
        # self.menuClasses.addAction(self.action_new_classes)

    def setup(self) -> None:
        """
        Sets the menus up
        """
        super().setup()
        self.class_names_group.triggered.connect(lambda x: self.parent().update_ROI_selection(x.text()))

    def set_class_names_choice(self) -> None:
        """
        Sets the options for class names, i.e. all lists of class names that are associated to annotation runs
        available in the current project.
        """
        self.menuClasses.clear()
        self.menuClasses.addAction(self.action_edit_classes)
        self.menuClasses.addSeparator()
        super().set_class_names_choice()
        self.action_edit_classes.triggered.connect(self.parent().define_classes)


class PredictionMenuBar(AnnotationMenuBar):
    """
    The menu bar for prediction viewers
    """

    def __init__(self, parent: PredictionViewer):
        super().__init__(parent)
        self.run_choice = []
        self.menu_prediction_runs = self.addMenu('Prediction runs')
        self.prediction_runs_group = QActionGroup(self)
        self.prediction_runs_group.setExclusive(True)
        self.prediction_runs_group.triggered.connect(lambda x: self.parent().select_prediction_run(x))

    def setup(self) -> None:
        """
        Sets the menus up
        """
        super().setup()
        self.set_run_choice(self.parent().plugin.class_names())
        self.class_names_group.triggered.connect(lambda x: self.set_run_choice(x.text()))

    def set_class_names_choice(self) -> None:
        """
        Sets the options for class names, i.e. all lists of class names that are associated to a classifier (trained or imported)
        available in the current project.
        """
        self.parent().plugin.update_class_names(prediction=True)
        self.menuClasses.clear()
        super().set_class_names_choice()

    def set_run_choice(self, class_names: str) -> None:
        """
        Sets the menu to choose the prediction run. Runs are selected according to the class names.
        By default, the last prediction run is selected.

        :param class_names: the class names to select prediction runs
        """
        self.parent().plugin.parameters['class_names'].set_value(class_names)
        self.run_choice = []
        for action in self.prediction_runs_group.actions():
            self.prediction_runs_group.removeAction(action)
        prediction_runs = self.parent().plugin.get_prediction_runs()
        for prediction_run in prediction_runs[class_names]:
            self.run_choice.append(QAction(f'run-{prediction_run}'))
            self.run_choice[-1].setCheckable(True)
            self.prediction_runs_group.addAction(self.run_choice[-1])
            self.menu_prediction_runs.addAction(self.run_choice[-1])
            self.run_choice[-1].setData(prediction_run)
        self.run_choice[-1].setChecked(True)


class DefineClassesDialog(Dialog):
    """
    Dialog window to define a set of classes for manual annotation
    """

    def __init__(self, annotator: ManualAnnotator, plugin: Plugin, suggestion: list[str] = None):
        super().__init__(plugin, title='Define classes')
        self.annotator = annotator
        self.list_view = ClassListView(self, multiselection=True)
        self.setup_class_names(suggestion)
        self.button_box = self.addButtonBox()

        self.import_class_btn = QPushButton('Import classes', parent=self.button_box)
        self.import_class_btn.clicked.connect(self.import_classes)
        self.button_box.addButton(self.import_class_btn, QDialogButtonBox.ButtonRole.ActionRole)

        self.add_class_btn = QPushButton('Add new class', parent=self.button_box)
        self.add_class_btn.clicked.connect(self.list_view.add_class)
        self.button_box.addButton(self.add_class_btn, QDialogButtonBox.ButtonRole.ActionRole)

        self.button_box.accepted.connect(self.save_new_classes)
        self.button_box.rejected.connect(self.close)

        self.arrangeWidgets([self.list_view, self.button_box])

        self.fit_to_contents()
        self.exec()

    def setup_class_names(self, suggestion: list[str] = None) -> None:
        """
        Sets the initial list of classes. If no suggestion is made, then the current list stored in the plugin is used. If none is
        available, then ['A', 'B'] is proposed. Otherwise, the suggestion made is used.

        :param suggestion: the class list suggestion
        """
        if suggestion is None:
            suggestion = self.plugin.class_names(as_string=False)
        if suggestion is None:
            self.list_view.model().setStringList(['A', 'B'])
        else:
            self.list_view.model().setStringList(suggestion)

    def save_new_classes(self) -> None:
        """
        Saves the new list of classes in the current annotation Run table
        """
        self.plugin.parameters["class_names"].add_item(
                {json.dumps(self.list_view.model().stringList()): self.list_view.model().stringList()})
        self.plugin.parameters["class_names"].set_value(self.list_view.model().stringList())
        if self.annotator.menubar is None:
            self.annotator.setup(plugin=self.plugin)
        self.annotator.save_run()
        self.annotator.menubar.set_class_names_choice()
        if self.annotator.roi_list:
            self.plugin.resume_manual_annotation(self.annotator, run=self.annotator.run,
                                                 roi_selection=self.annotator.roi_list.data)
        else:
            self.plugin.resume_manual_annotation(self.annotator, run=self.annotator.run)
        self.annotator.update_roi_classes_plot()
        self.close()

    def import_classes(self) -> None:
        """
        Selects a csv file containing ROI frames annotations and open a FOV2ROIlinks window to load the data it contains
        into the database as FOVs and ROIs with annotations.
        """
        filters = ["txt (*.txt)", ]
        init_dir = get_config_value('project', 'workspace')
        class_names_file, _ = QFileDialog.getOpenFileName(PyDetecDiv.main_window,
                                                          caption='Choose text file with class names',
                                                          dir=init_dir,
                                                          filter=";;".join(filters),
                                                          selectedFilter=filters[0])
        if class_names_file:
            self.setup_class_names(suggestion=sorted([line.strip() for line in open(class_names_file, 'r')]))


class ClassListView(ListView):
    """
    A list view used to define a set of classes for annotation
    """

    def __init__(self, parent: DefineClassesDialog, parameter: Parameter = None, height: int = None, multiselection: bool = False,
                 **kwargs: dict[str, Any]):
        super().__init__(parent, parameter=parameter, height=height, multiselection=multiselection, **kwargs)

    def contextMenuEvent(self, e: QContextMenuEvent) -> None:
        """
        Defines a contextual menu

        :param e: the contextual menu event
        """
        if self.model().rowCount():
            context = QMenu(self)
            add_class = QAction("Add a new class", self)
            add_class.triggered.connect(self.add_class)
            context.addAction(add_class)
            unselect = QAction("Unselect all", self)
            unselect.triggered.connect(self.unselect)
            context.addAction(unselect)
            toggle = QAction("Toggle selection", self)
            toggle.triggered.connect(self.toggle)
            context.addAction(toggle)
            context.addSeparator()
            remove = QAction("Remove selected items", self)
            remove.triggered.connect(self.remove_items)
            context.addAction(remove)
            clear_list = QAction("Clear list", self)
            context.addAction(clear_list)
            clear_list.triggered.connect(self.clear_list)
            context.exec(e.globalPos())

    def add_class(self) -> None:
        """
        Adds a new line to the current ListView model and selects it for the user to type the new class name in
        """
        new_list = self.model().stringList() + ['']
        self.model().setStringList(new_list)
        self.unselect()
        selection = self.model().index(self.model().rowCount() - 1, 0)
        self.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.Select)
        self.edit(selection)
