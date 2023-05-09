import os
import time
import pandas as pd
from PySide6.QtCore import Signal, Qt, QRect, QPoint, QTimer
from PySide6.QtGui import QPixmap, QImage, QPen, QTransform, QKeySequence
from PySide6.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QFileDialog, QMenu
import numpy as np
import cv2 as cv
from skimage.feature import peak_local_max

from pydetecdiv.app import WaitDialog, PyDetecDiv, DrawingTools, pydetecdiv_project
from pydetecdiv.app.gui.ui.ImageViewer import Ui_ImageViewer
from pydetecdiv.domain.ImageResource import ImageResource
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.settings import get_config_value
from pydetecdiv.utils import round_to_even


class ImageViewer(QMainWindow, Ui_ImageViewer):
    video_frame = Signal(int)
    finished = Signal(bool)

    def __init__(self, **kwargs):
        QMainWindow.__init__(self)
        self.ui = Ui_ImageViewer()
        self.setWindowTitle('Image viewer')
        self.ui.setupUi(self)
        self.scale = 100
        self.ui.z_slider.setEnabled(False)
        self.ui.t_slider.setEnabled(False)
        self.ui.z_slider.setPageStep(1)
        self.image_resource = None
        self.scene = ViewerScene()
        self.scene.setParent(self)
        self.pixmap = QPixmap()
        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.ui.viewer.setScene(self.scene)
        self.fov = None
        self.stage = None
        self.project_name = None
        self.C = 0
        self.T = 0
        self.Z = 0
        # self.drift = None
        self.apply_drift = False
        self.roi_template = None
        self.video_playing = False
        self.video_frame.emit(self.T)
        self.video_frame.connect(lambda frame: self.ui.current_frame.setText(f'Frame: {frame}'))
        self.video_frame.connect(self.ui.t_slider.setSliderPosition)
        self.crop = None
        self.timer = None
        self.start = None
        self.first_frame = None
        self.frame = None
        self.wait = None

    def set_image_resource(self, image_resource, crop=None):
        self.image_resource = image_resource
        self.T, self.C, self.Z = (0, 0, 0)

        self.ui.view_name.setText(f'View: {image_resource.fov.name}')
        drift_filename = os.path.join(get_config_value('project', 'workspace'), PyDetecDiv().project_name,
                                      f'{self.image_resource.fov.name}_drift.csv')
        if os.path.isfile(drift_filename):
            self.parent().parent().drift = pd.read_csv(drift_filename)
            self.ui.actionPlot.setEnabled(True)
            self.ui.actionApply_correction.setEnabled(True)
        else:
            self.parent().parent().drift = None
        self.crop = crop

        self.ui.z_slider.setMinimum(0)
        self.ui.z_slider.setMaximum(image_resource.sizeZ - 1)
        self.ui.z_slider.setEnabled(True)

        self.ui.t_slider.setMinimum(0)
        self.ui.t_slider.setMaximum(image_resource.sizeT - 1)
        self.ui.t_slider.setEnabled(True)

    def set_channel(self, C):
        self.C = C

    def zoom_reset(self):
        self.ui.viewer.scale(100 / self.scale, 100 / self.scale)
        self.scale = 100
        self.ui.zoom_value.setSliderPosition(100)
        self.ui.scale_value.setText(f'Zoom: {self.scale}%')

    def zoom_fit(self):
        self.ui.viewer.fitInView(self.pixmapItem, Qt.KeepAspectRatio)
        self.scale = int(100 * np.around(self.ui.viewer.transform().m11(), 2))
        self.ui.zoom_value.setSliderPosition(self.scale)
        self.ui.scale_value.setText(f'Zoom: {self.scale}%')

    def zoom_set_value(self, value):
        self.ui.viewer.scale(value / self.scale, value / self.scale)
        self.scale = value
        self.ui.scale_value.setText(f'Zoom: {self.scale}%')

    def play_video(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_next_frame)
        self.timer.setInterval(50)
        self.start = time.time()
        self.video_playing = True
        self.first_frame = self.T
        self.frame = self.T
        self.timer.start()

    def show_next_frame(self):
        self.frame += 1
        if self.frame >= self.image_resource.sizeT or not self.video_playing:
            self.timer.stop()
        else:
            end = time.time()
            speed = np.around((self.frame - self.first_frame) / (end - self.start), 1)
            self.ui.FPS.setText(f'FPS: {speed}')
            self.change_frame(self.frame)

    def pause_video(self):
        self.video_playing = False

    def video_back(self):
        self.change_frame(T=0)

    def change_layer(self, Z=0):
        if self.Z != Z:
            self.Z = Z
            self.display()

    def change_frame(self, T=0):
        if self.T != T:
            self.T = T
            self.video_frame.emit(self.T)
            self.display()

    def display(self, C=None, T=None, Z=None):
        C = self.C if C is None else C
        T = self.T if T is None else T
        Z = self.Z if Z is None else Z
        if self.apply_drift:
            idx = T if T < len(self.parent().parent().drift) else T - 1
            arr = self.image_resource.image(C=C, T=T, Z=Z, drift=self.parent().parent().drift.iloc[idx])
        else:
            arr = self.image_resource.image(C=C, T=T, Z=Z)
        print(self.crop)
        if self.crop is not None:
            arr = arr[..., self.crop[1], self.crop[0]]
            print(f'cropping to {self.crop}')
        ny, nx = arr.shape
        print(f'display shape: {arr.shape}')
        img = QImage(np.ascontiguousarray(arr.data), nx, ny, QImage.Format_Grayscale16)
        self.pixmap.convertFromImage(img)
        self.pixmapItem.setPixmap(self.pixmap)

    def draw_saved_rois(self, roi_list):
        for roi in roi_list:
            rect_item = self.scene.addRect(QRect(0, 0, roi.width, roi.height))
            rect_item.setPen(self.scene.saved_pen)
            rect_item.setPos(QPoint(roi.x, roi.y))
            rect_item.setFlags(QGraphicsItem.ItemIsSelectable)
            rect_item.setData(0, roi.name)

    def close_window(self):
        self.parent().parent().window.close()

    def compute_drift(self):
        self.wait = WaitDialog('Computing drift, please wait.', self, cancel_msg='Cancel drift computation please wait')
        self.finished.connect(self.wait.close_window)
        self.wait.wait_for(self.compute_and_plot_drift)
        self.plot_drift()

    def plot_drift(self):
        self.parent().parent().show_plot(self.parent().parent().drift, 'Drift')
        for viewer in self.parent().parent().get_image_viewers():
            viewer.ui.actionPlot.setEnabled(True)
            viewer.ui.actionApply_correction.setEnabled(True)

    def compute_and_plot_drift(self):
        self.parent().parent().drift = self.image_resource.compute_drift(Z=self.Z, C=self.C, method='vidstab')
        self.finished.emit(True)

    def apply_drift_correction(self):
        if self.ui.actionApply_correction.isChecked():
            drift_filename = os.path.join(get_config_value('project', 'workspace'), PyDetecDiv().project_name,
                                          f'{self.image_resource.fov.name}_drift.csv')
            self.parent().parent().drift.to_csv(drift_filename, index=False)
        self.apply_drift = self.ui.actionApply_correction.isChecked()
        self.display()

    def set_roi_template(self):
        roi = self.scene.get_selected_ROI()
        if roi:
            data = self.get_roi_image(roi)
            self.roi_template = np.uint8(np.array(data) / np.max(data) * 255)
            self.ui.actionIdentify_ROIs.setEnabled(True)
            self.ui.actionView_template.setEnabled(True)

    def get_roi_image(self, roi):
        w, h = roi.rect().toRect().size().toTuple()
        pos = roi.pos()
        x1, x2 = int(pos.x()), w + int(pos.x())
        y1, y2 = int(pos.y()), h + int(pos.y())
        return self.image_resource.image(C=self.C, T=self.T, Z=self.Z)[y1:y2, x1:x2]

    def get_roi_data(self, roi):
        w, h = roi.rect().toRect().size().toTuple()
        pos = roi.pos()
        x1, x2 = int(pos.x()), w + int(pos.x())
        y1, y2 = int(pos.y()), h + int(pos.y())
        crop = None
        if self.parent().parent().drift is not None:
            max_shift_x = np.max(np.abs(self.parent().parent().drift.dx))
            max_shift_y = np.max(np.abs(self.parent().parent().drift.dy))
            x1, x2 = round_to_even(x1 - max_shift_x, ceil=False), round_to_even(x2 + max_shift_x)
            y1, y2 = round_to_even(y1 - max_shift_y, ceil=False), round_to_even(y2 + max_shift_y)
            x, y = round_to_even(max_shift_x, ceil=False), round_to_even(max_shift_y, ceil=False)
            crop = [slice(x, x + w), slice(y, y + h)]
            # min_shift_x, max_shift_x = np.min(self.drift.dx), np.max(self.drift.dx)
            # min_shift_y, max_shift_y = np.min(self.drift.dy), np.max(self.drift.dy)
            # print(min_shift_x, max_shift_x, min_shift_y, max_shift_y)
            # print(f'without margins: ({x1}, {y1}) - ({x2}, {y2}) [{x2 - x1}, {y2 - y1}]')
            # x1 = round_to_even(np.min([x1 + min_shift_x, x1]), ceil=False)
            # x2 = round_to_even(np.max([x2 + max_shift_x, x2]))
            # y1 = round_to_even(np.min([y1 + min_shift_y, y1]), ceil=False)
            # y2 = round_to_even(np.max([y2 + max_shift_y, y2]))
            # print(f'   with margins: ({x1}, {y1}) - ({x2}, {y2}) [{x2 - x1}, {y2 - y1}]')
            # x = round_to_even(max_shift_x, ceil=False) if max_shift_x > 0 else 0
            # y = round_to_even(max_shift_y, ceil=False) if max_shift_y > 0 else 0
            # crop = [slice(x, x+w), slice(y, y+h)]
            # print(f'should crop to {x}, {y}, {x+w}, {y+h} - {self.crop} - {w}, {h}')
        return self.image_resource.data_sample(X=slice(x1, x2), Y=slice(y1, y2)), crop

    def load_roi_template(self):
        filename = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.tif *.tiff)")[0]
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        self.roi_template = np.uint8(np.array(img / np.max(img) * 255))
        self.ui.actionIdentify_ROIs.setEnabled(True)

    def identify_rois(self):
        threshold = 0.3
        img = self.image_resource.image(C=self.C, Z=self.Z, T=self.T)
        img8bits = np.uint8(np.array(img / np.max(img) * 255))
        res = cv.matchTemplate(img8bits, self.roi_template, cv.TM_CCOEFF_NORMED)
        # loc = np.where(res >= threshold)
        # xy = peak_local_max(res,min_distance=self.roi_template.shape[0],threshold_abs=threshold,exclude_border=False)
        xy = peak_local_max(res, threshold_abs=threshold, exclude_border=False)
        w, h = self.roi_template.shape[::-1]
        for pt in xy:
            x, y = pt[1], pt[0]
            if not isinstance(self.scene.itemAt(QPoint(x, y), QTransform().scale(1, 1)), QGraphicsRectItem):
                rect_item = self.scene.addRect(QRect(0, 0, w, h))
                rect_item.setPos(x, y)
                if [r for r in rect_item.collidingItems(Qt.IntersectsItemBoundingRect) if
                    isinstance(r, QGraphicsRectItem)]:
                    self.scene.removeItem(rect_item)
                else:
                    rect_item.setPen(self.scene.match_pen)
                    rect_item.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)

    def view_template(self):
        self.parent().parent().show_image(self.roi_template, title='ROI template', format_=QImage.Format_Grayscale8)

    def view_roi_image(self, selected_roi=None):
        if selected_roi is None:
            selected_roi = self.scene.get_selected_ROI()
        viewer = ImageViewer()
        data, crop = self.get_roi_data(selected_roi)
        self.parent().parent().addTab(viewer, selected_roi.data(0))
        viewer.set_image_resource(ImageResource(data=data, fov=self.image_resource.fov), crop=crop)
        viewer.ui.view_name.setText(f'View: {selected_roi.data(0)}')
        viewer.display()
        self.parent().parent().setCurrentWidget(viewer)

    def save_rois(self):
        rois = [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem)]
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            for i, rect_item in enumerate(sorted(rois, key=lambda x: x.scenePos().toPoint().toTuple())):
                x, y = rect_item.scenePos().toPoint().toTuple()
                w, h = rect_item.rect().toRect().getCoords()[2:]
                new_roi = ROI(project=project, name=f'{self.image_resource.fov.name}_{i}',
                              fov=self.image_resource.fov, top_left=(x, y), bottom_right=(int(x) + w, int(y) + h))
                rect_item.setData(0, new_roi.name)
        self.fixate_saved_rois()

    def fixate_saved_rois(self):
        for r in [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem)]:
            r.setPen(self.scene.saved_pen)
            r.setFlag(QGraphicsItem.ItemIsMovable, False)


class ViewerScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.from_x = None
        self.from_y = None
        self.pen = QPen(Qt.GlobalColor.cyan, 2)
        self.match_pen = QPen(Qt.GlobalColor.yellow, 2)
        self.saved_pen = QPen(Qt.GlobalColor.green, 2)
        self.warning_pen = QPen(Qt.GlobalColor.red, 2)

    def contextMenuEvent(self, event):
        menu = QMenu()
        view_in_new_tab = menu.addAction("View in new tab")
        r = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
        if isinstance(r, QGraphicsRectItem):
            view_in_new_tab.triggered.connect(lambda _: self.parent().view_roi_image(r))
        selectedAction = menu.exec(event.screenPos())

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Delete):
            for r in self.selectedItems():
                self.removeItem(r)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            match PyDetecDiv().current_drawing_tool:
                case DrawingTools.Cursor:
                    self.select_ROI(event)
                case DrawingTools.DrawROI:
                    self.select_ROI(event)
                case DrawingTools.DuplicateROI:
                    self.duplicate_selected_ROI(event)

    def select_ROI(self, event):
        _ = [r.setSelected(False) for r in self.items()]
        r = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
        if isinstance(r, QGraphicsRectItem):
            r.setSelected(True)
        if self.selectedItems():
            self.parent().ui.actionSet_template.setEnabled(True)
        else:
            self.parent().ui.actionSet_template.setEnabled(False)

    def get_selected_ROI(self):
        for selection in self.selectedItems():
            if isinstance(selection, QGraphicsRectItem):
                return selection
        return None

    def duplicate_selected_ROI(self, event):
        pos = event.scenePos()
        roi = self.get_selected_ROI()
        if roi:
            roi = self.addRect(roi.rect())
            roi.setPen(self.pen)
            w, h = roi.rect().size().toTuple()
            roi.setPos(QPoint(pos.x() - np.around(w / 2.0), pos.y() - np.around(h / 2.0)))
            roi.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
            roi.setData(0, f'Region{len(self.items())}')
            self.select_ROI(event)
            if [r for r in roi.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
                roi.setPen(self.warning_pen)
            else:
                roi.setPen(self.pen)

    def mouseMoveEvent(self, event):
        if event.button() == Qt.NoButton:
            match PyDetecDiv().current_drawing_tool, event.modifiers():
                case DrawingTools.Cursor, Qt.NoModifier:
                    self.move_ROI(event)
                case DrawingTools.Cursor, Qt.ControlModifier:
                    self.draw_ROI(event)
                case DrawingTools.DrawROI, Qt.NoModifier:
                    self.draw_ROI(event)
                case DrawingTools.DrawROI, Qt.ControlModifier:
                    self.move_ROI(event)
                case DrawingTools.DuplicateROI, Qt.NoModifier:
                    self.move_ROI(event)

    def move_ROI(self, event):
        roi = self.get_selected_ROI()
        if roi and (roi.flags() & QGraphicsItem.ItemIsMovable):
            pos = event.scenePos()
            roi.moveBy(pos.x() - event.lastScenePos().x(), pos.y() - event.lastScenePos().y())
            if [r for r in roi.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
                roi.setPen(self.warning_pen)
            else:
                roi.setPen(self.pen)

    def draw_ROI(self, event):
        roi = self.get_selected_ROI()
        pos = event.scenePos()
        if roi and (roi.flags() & QGraphicsItem.ItemIsMovable):
            roi_pos = roi.scenePos()
            w, h = round_to_even(pos.x() - roi_pos.x()), round_to_even(pos.y() - roi_pos.y())
            rect = QRect(0, 0, w, h)
            roi.setRect(rect)
        else:
            roi = self.addRect(QRect(0, 0, 2, 2))
            roi.setPen(self.pen)
            roi.setPos(QPoint(pos.x(), pos.y()))
            roi.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
            roi.setData(0, f'Region{len(self.items())}')
            self.select_ROI(event)
        if [r for r in roi.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
            roi.setPen(self.warning_pen)
        else:
            roi.setPen(self.pen)
