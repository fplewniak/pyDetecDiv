from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QCursor, QImage, QKeySequence
from PySide6.QtWidgets import QGraphicsScene, QGraphicsTextItem
import numpy as np

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.ImageViewer import ImageViewer, ViewerScene


def open_annotator_tab(selected_roi, scene):
    project_window = scene.parent()
    viewer = Annotator()
    viewer.ui.zoom_value.setMaximum(400)
    PyDetecDiv().setOverrideCursor(QCursor(Qt.WaitCursor))
    viewer.image_source_ref = selected_roi if selected_roi else project_window.scene.get_selected_ROI()
    viewer.parent_viewer = project_window
    data, crop = project_window.get_roi_data(viewer.image_source_ref)
    project_window.parent().parent().addTab(viewer, viewer.image_source_ref.data(0))
    # viewer.set_image_resource_data(ArrayImageResource(data=data, fov=self.image_resource_data.fov, image_resource=self.image_resource_data.image_resource), crop=crop)
    viewer.set_image_resource_data(project_window.image_resource_data, crop=crop)
    viewer.ui.view_name.setText(f'View: {viewer.image_source_ref.data(0)}')
    viewer.synchronize_with(project_window)
    viewer.display()
    project_window.parent().parent().setCurrentWidget(viewer)
    PyDetecDiv().restoreOverrideCursor()


class Annotator(ImageViewer):
    def __init__(self):
        super().__init__()
        self.scene = AnnotatorScene()
        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.ui.viewer.setScene(self.scene)
        self.scene.setParent(self)
        self.viewport_rect = None
        self.zoom_set_value(200)

    def display(self, C=None, T=None, Z=None, class_name=' - '):
        super().display(C, T, Z)
        class_item = QGraphicsTextItem(class_name)
        text_boundingRect = class_item.boundingRect()
        frame_boundingRect = self.pixmapItem.boundingRect()
        class_item.setPos((frame_boundingRect.width() - text_boundingRect.width())/2 , frame_boundingRect.height())
        self.scene.addItem(class_item)
        self.viewport_rect = QRectF(min([class_item.x(), self.pixmapItem.x()]),
                                    min([class_item.y(), self.pixmapItem.y()]) - 5,
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

    def keyPressEvent(self, event):
        print(event)
        if event.text() == '1':
            self.parent().display(class_name='class 1')
        elif event.matches(QKeySequence.MoveToNextChar):
            self.parent().change_frame(self.parent().T + 1)
        elif event.matches(QKeySequence.MoveToPreviousChar):
            self.parent().change_frame(self.parent().T - 1)
