"""
ROI annotation for image classification
"""
import sqlalchemy
from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import QGraphicsTextItem, QGraphicsScene
import numpy as np

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.ImageViewer import ImageViewer
from pydetecdiv.settings import get_config_value


def open_annotator(plugin, roi_selection):
    """
    Open an annotator instance with a selection of ROIs
    :param plugin: the plugin instance
    :param roi_selection: the list of ROIs to annotate
    """
    project_window = PyDetecDiv().main_window.active_subwindow
    viewer = Annotator()
    viewer.set_plugin(plugin)
    viewer.ui.zoom_value.setMaximum(400)
    project_window.addTab(viewer, 'ROI annotation')
    viewer.set_roi_list(roi_selection)
    viewer.next_roi()


class Annotator(ImageViewer):
    """
    Annotator class extending the ImageViewer class to define functionalities specific to ROI image annotation
    """

    def __init__(self):
        super().__init__()
        self.roi_list = None
        self.roi = None
        self.run = None
        self.setObjectName('Annotator')
        self.scene = AnnotatorScene()
        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.ui.viewer.setScene(self.scene)
        self.scene.setParent(self)
        self.viewport_rect = None
        self.zoom_set_value(200)
        self.class_item = QGraphicsTextItem('-')
        self.roi_classes = []
        self.plugin = None
        self.scene.addItem(self.class_item)

    def set_plugin(self, plugin):
        """
        Define the plugin instance to enable the annotator to access some data
        :param plugin: the plugin instance
        """
        self.plugin = plugin

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
            with pydetecdiv_project(PyDetecDiv().project_name) as project:
                image_resource = project.get_linked_objects('FOV', self.roi)[0].image_resource()
                x1, x2 = self.roi.top_left[0], self.roi.bottom_right[0] + 1
                y1, y2 = self.roi.top_left[1], self.roi.bottom_right[1] + 1
                crop = (slice(x1, x2), slice(y1, y2))
                self.set_image_resource_data(image_resource.image_resource_data(), crop=crop)
                self.display()
                self.get_roi_annotations()
                self.change_frame(0)
                self.video_frame.emit(0)
                self.ui.t_slider.setSliderPosition(0)
                self.ui.view_name.setText(f'ROI: {self.roi.name}')
                PyDetecDiv().main_window.active_subwindow.setCurrentWidget(self)
        except StopIteration:
            pass

    def get_roi_annotations(self):
        """
        Retrieve from the database the manual annotations for a ROI
        """
        self.roi_classes = ['-'] * self.image_resource_data.sizeT
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            results = list(project.repository.session.execute(
                sqlalchemy.text(f"SELECT rc.roi,rc.t,rc.class_name,run.parameters ->> '$.annotator' as annotator "
                                f"FROM run, roi_classification as rc "
                                f"WHERE run.command='annotate_rois' and rc.run=run.id_ and rc.roi={self.roi.id_} "
                                f"AND annotator='{get_config_value('project', 'user')}' "
                                f"ORDER BY rc.run ASC;")))
            for annotation in results:
                self.roi_classes[annotation[1]] = annotation[2]

    def annotate_current(self, class_name=None):
        """
        Assign the class name to the current frame
        :param class_name: the class name
        """
        self.roi_classes[self.T] = class_name

    def display_class_name(self, roi_class=None):
        """
        Display the class name below the frame
        :param roi_class: the class name
        """
        if self.roi_classes[self.T] == '-' and self.T > 0 and self.roi_classes[self.T - 1] != '-':
            roi_class = self.roi_classes[self.T - 1]
            self.class_item.setDefaultTextColor('red')
        else:
            roi_class = self.roi_classes[self.T] if roi_class is None else roi_class
            self.class_item.setDefaultTextColor('black')
        self.class_item.setPlainText(roi_class)
        text_boundingRect = self.class_item.boundingRect()
        frame_boundingRect = self.pixmapItem.boundingRect()
        self.class_item.setPos((frame_boundingRect.width() - text_boundingRect.width()) / 2,
                               frame_boundingRect.height())
        self.viewport_rect = QRectF(min([self.class_item.x(), self.pixmapItem.x()]),
                                    min([self.class_item.y(), self.pixmapItem.y()]) - 5,
                                    max([text_boundingRect.width(), frame_boundingRect.width()]),
                                    text_boundingRect.height() + frame_boundingRect.height() + 5,
                                    )

    def zoom_fit(self):
        """
        Set the zoom value to fit the image in the viewer
        """
        self.ui.viewer.fitInView(self.viewport_rect, Qt.KeepAspectRatio)
        self.scale = int(100 * np.around(self.ui.viewer.transform().m11(), 2))
        self.ui.zoom_value.setSliderPosition(self.scale)
        self.ui.scale_value.setText(f'Zoom: {self.scale}%')

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
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            self.run = self.plugin.save_run(project, 'annotate_rois',
                                            {'class_names': self.plugin.class_names,
                                             'annotator': get_config_value('project', 'user'),
                                             'roi_num': self.plugin.gui.roi_number.value(),
                                             })


class AnnotatorScene(QGraphicsScene):
    """
    The viewer scene where images and other items are drawn
    """

    def focusInEvent(self, event):
        """
        When the scene is in focus, then draw a larger frame around it to indicate its in-focus status
        :param event: the focusInEvent
        """
        self.parent().ui.viewer.setLineWidth(3)

    def focusOutEvent(self, event):
        """
        When the scene is out of focus, then draw a small frame around it to indicate its out-focus status
        :param event: the focusOutEvent
        """
        self.parent().ui.viewer.setLineWidth(1)

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
        if event.text() in list('azertyuiop')[0:len(self.parent().plugin.class_names)]:
            self.parent().annotate_current(self.parent().plugin.class_names["azertyuiop".find(event.text())])
            self.parent().change_frame(min(self.parent().T + 1, self.parent().image_resource_data.sizeT - 1))
        elif event.text() == ' ':
            self.parent().annotate_current(class_name=f'{self.parent().class_item.toPlainText()}')
            self.parent().change_frame(min(self.parent().T + 1, self.parent().image_resource_data.sizeT - 1))
        elif event.key() == Qt.Key_Right:
            self.parent().change_frame(min(self.parent().T + 1, self.parent().image_resource_data.sizeT - 1))
        elif event.key() == Qt.Key_Left:
            self.parent().change_frame(max(self.parent().T - 1, 0))
        elif event.key() == Qt.Key_Enter:
            self.parent().annotate_current(class_name=f'{self.parent().class_item.toPlainText()}')
            self.parent().class_item.setDefaultTextColor('black')
            if self.parent().run is None:
                self.parent().save_run()
            self.parent().plugin.save_annotations(self.parent().roi, self.parent().roi_classes, self.parent().run)
            self.parent().next_roi()
        elif event.key() == Qt.Key_Escape:
            self.parent().next_roi()
