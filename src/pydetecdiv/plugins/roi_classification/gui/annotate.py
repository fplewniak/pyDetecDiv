"""
ROI annotation for image classification
"""

from PySide6.QtCore import Qt, QRectF, QEvent
from PySide6.QtWidgets import QGraphicsTextItem, QGraphicsScene
import numpy as np

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.ImageViewer import ImageViewer


def open_annotator_from_selection(plugin, selected_roi, scene):
    """
    Open an annotator instance to annotate a ROI selected in the Image viewer.
    :param plugin: the Plugin instance
    :param selected_roi: the selected ROI
    :param scene: the ViewerScene instance
    """
    project_window = scene.parent()
    viewer = Annotator()
    viewer.set_plugin(plugin)
    viewer.ui.zoom_value.setMaximum(400)
    viewer.image_source_ref = selected_roi if selected_roi else project_window.scene.get_selected_ROI()
    viewer.parent_viewer = project_window
    _, crop = project_window.get_roi_data(viewer.image_source_ref)
    project_window.parent().parent().addTab(viewer, viewer.image_source_ref.data(0))
    viewer.set_image_resource_data(project_window.image_resource_data, crop=crop)
    viewer.roi_classes = ['-'] * viewer.image_resource_data.sizeT
    viewer.ui.view_name.setText(f'View: {viewer.image_source_ref.data(0)}')
    viewer.synchronize_with(project_window)
    viewer.display()
    project_window.parent().parent().setCurrentWidget(viewer)


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
    roi = roi_selection[0]
    project_window.addTab(viewer, roi.name)
    with pydetecdiv_project(PyDetecDiv().project_name) as project:
        image_resource = project.get_linked_objects('FOV', roi)[0].image_resource()
        x1, x2 = roi.top_left[0], roi.bottom_right[0] + 1
        y1, y2 = roi.top_left[1], roi.bottom_right[1] + 1
        crop = (slice(x1, x2), slice(y1, y2))
        viewer.set_image_resource_data(image_resource.image_resource_data(), crop=crop)
        viewer.roi_classes = ['-'] * viewer.image_resource_data.sizeT
        viewer.display()
        viewer.display_class_name('-')
        project_window.setCurrentWidget(viewer)


class Annotator(ImageViewer):
    """
    Annotator class extending the ImageViewer class to define functionalities specific to ROI image annotation
    """
    def __init__(self):
        super().__init__()
        self.setObjectName('Annotator')
        self.scene = AnnotatorScene()
        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.ui.viewer.setScene(self.scene)
        self.scene.setParent(self)
        self.scene.installEventFilter(self)
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

    def change_frame(self, T: object = 0):
        """

        :param T:
        """
        super().change_frame(T)
        self.display_class_name()

    def eventFilter(self, watched, event):
        """
        Event filter to handle events related to the AnnotatorScene (focus, annotation, viewing next or previous frame)
        :param watched: the watched object
        :param event: the caught event
        :return: True if an event was captured and should not be forwarded, False otherwise.
        """
        if watched == self.scene:
            if event.type() == QEvent.ShortcutOverride:
                if event.text() in list('azertyuiop')[0:len(self.plugin.class_names)]:
                    self.annotate_current(self.plugin.class_names["azertyuiop".find(event.text())])
                    self.change_frame(min(self.T + 1, self.image_resource_data.sizeT - 1))
                elif event.text() == ' ':
                    self.annotate_current(class_name=f'{self.class_item.toPlainText()}')
                    self.change_frame(min(self.T + 1, self.image_resource_data.sizeT - 1))
                    return True
                elif event.key() == Qt.Key_Right:
                    self.change_frame(min(self.T + 1, self.image_resource_data.sizeT - 1))
                    return True
                elif event.key() == Qt.Key_Left:
                    self.change_frame(max(self.T - 1, 0))
                    return True
            elif event.type() == QEvent.FocusIn:
                self.ui.viewer.setLineWidth(3)
            elif event.type() == QEvent.FocusOut:
                self.ui.viewer.setLineWidth(1)
        return False


class AnnotatorScene(QGraphicsScene):
    """
    The viewer scene where images and other items are drawn
    """
