from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QCursor, QImage, QKeySequence
from PySide6.QtWidgets import QGraphicsScene, QGraphicsTextItem, QDockWidget
import numpy as np

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.ImageViewer import ImageViewer, ViewerScene


def open_annotator_from_selection(plugin, selected_roi, scene):
    project_window = scene.parent()
    viewer = Annotator()
    viewer.set_plugin(plugin)
    viewer.ui.zoom_value.setMaximum(400)
    PyDetecDiv().setOverrideCursor(QCursor(Qt.WaitCursor))
    viewer.image_source_ref = selected_roi if selected_roi else project_window.scene.get_selected_ROI()
    viewer.parent_viewer = project_window
    data, crop = project_window.get_roi_data(viewer.image_source_ref)
    project_window.parent().parent().addTab(viewer, viewer.image_source_ref.data(0))
    viewer.set_image_resource_data(project_window.image_resource_data, crop=crop)
    viewer.roi_classes = ['-'] * viewer.image_resource_data.sizeT
    viewer.ui.view_name.setText(f'View: {viewer.image_source_ref.data(0)}')
    viewer.synchronize_with(project_window)
    viewer.display()
    project_window.parent().parent().setCurrentWidget(viewer)
    PyDetecDiv().restoreOverrideCursor()


def open_annotator(plugin, roi_selection):
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
        project_window.setCurrentWidget(viewer)


class Annotator(ImageViewer):
    def __init__(self):
        super().__init__()
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
        self.plugin = plugin
        self.scene.plugin = plugin

    def annotate_current(self, class_name='-'):
        self.roi_classes[self.T] = class_name
        self.display()

    def display(self, C=None, T=None, Z=None):
        super().display(C, T, Z)
        self.display_class_name()

    def display_class_name(self):
        if self.roi_classes[self.T] == '-' and self.T > 0 and self.roi_classes[self.T - 1] != '-':
            self.roi_classes[self.T] = self.roi_classes[self.T - 1]
            self.class_item.setDefaultTextColor('red')
        else:
            self.class_item.setDefaultTextColor('black')
        self.class_item.setPlainText(self.roi_classes[self.T])
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


class AnnotatorScene(ViewerScene):
    """
    The viewer scene where images and other items are drawn
    """

    def __init__(self):
        super().__init__()
        self.plugin = None

    def keyPressEvent(self, event):
        if event.text() in ['a', 'z', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'][0:len(self.plugin.class_names)]:
            self.parent().annotate_current(class_name=f'{self.plugin.class_names["azertyuiop".find(event.text())]}')
        elif event.matches(QKeySequence.MoveToNextChar):
            self.parent().change_frame(min(self.parent().T + 1, self.parent().image_resource_data.sizeT - 1))
        elif event.matches(QKeySequence.MoveToPreviousChar):
            self.parent().change_frame(max(self.parent().T - 1, 0))
