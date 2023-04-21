from PySide6.QtCore import Signal, Qt, QRect, QPoint
from PySide6.QtGui import QPixmap, QImage, QPen, QTransform, QKeySequence
from PySide6.QtWidgets import QMainWindow, QGraphicsScene, QApplication, QGraphicsItem, QGraphicsRectItem, QFileDialog
import time
import numpy as np
import cv2 as cv
from skimage.feature import peak_local_max

from pydetecdiv.app import WaitDialog, PyDetecDiv, DrawingTools
from pydetecdiv.app.gui.ui.ImageViewer import Ui_ImageViewer
from pydetecdiv.domain.ImageResource import ImageResource


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
        self.drift = None
        self.roi_template = None
        self.video_playing = False
        self.video_frame.emit(self.T)
        self.video_frame.connect(lambda frame: self.ui.current_frame.setText(f'Frame: {frame}'))
        self.video_frame.connect(self.ui.t_slider.setSliderPosition)

    def set_image_resource(self, image_resource):
        self.image_resource = image_resource
        self.T, self.C, self.Z = (0, 0, 0)

        self.ui.view_name.setText(f'View: {image_resource.fov.name}')

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

    # def play_video(self):
    #     threadCount = QThreadPool.globalInstance().maxThreadCount()
    #     pool = QThreadPool.globalInstance()
    #     for i in range(threadCount):
    #         runnable = Video_Player(self)
    #         pool.start(runnable)

    def play_video(self):
        start = time.time()
        self.video_playing = True
        first_frame = self.T
        for frame in range(first_frame, self.image_resource.sizeT):
            end = time.time()
            speed = np.around((frame - first_frame) / (end - start), 1)
            self.ui.FPS.setText(f'FPS: {speed}')
            QApplication.processEvents()
            self.change_frame(frame)
            if not self.video_playing:
                break

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
        arr = self.image_resource.image(C=C, T=T, Z=Z)
        ny, nx = arr.shape
        img = QImage(arr.data, nx, ny, QImage.Format_Grayscale16)
        self.pixmap.convertFromImage(img)
        self.pixmapItem.setPixmap(self.pixmap)

    def close_window(self):
        self.parent().parent().window.close()

    def compute_drift(self):
        self.wait = WaitDialog('Computing drift, please wait.', self, cancel_msg='Cancel drift computation please wait')
        self.finished.connect(self.wait.close_window)
        self.wait.wait_for(self.compute_and_plot_drift)
        self.plot_drift()

    def plot_drift(self):
        self.parent().parent().show_plot(self.drift, 'Drift')
        self.ui.actionPlot.setEnabled(True)

    def compute_and_plot_drift(self):
        self.drift = self.image_resource.compute_drift(Z=self.Z, C=self.C)
        self.finished.emit(True)

    def apply_drift_correction(self):
        print('Apply correction')

    def set_roi_template(self):
        coords = self.scene.rect_item.rect().getCoords()
        pos = self.scene.rect_item.pos()
        x1, x2 = int(coords[0] + pos.x()), int(coords[2] + pos.x())
        y1, y2 = int(coords[1] + pos.y()), int(coords[3] + pos.y())
        self.roi_template = np.uint8(np.array(self.image_resource.image()[x1:x2, y1:y2]) / 65535 * 255)
        self.ui.actionIdentify_ROIs.setEnabled(True)

    def load_roi_template(self):
        filename = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.tif *.tiff)")[0]
        self.roi_template = np.uint8(np.array(cv.imread(filename)))
        self.ui.actionIdentify_ROIs.setEnabled(True)

    def identify_rois(self):
        threshold = 0.3
        img8bits = np.uint8(np.array(self.image_resource.image(C=self.C, Z=self.Z, T=self.T) / 65535 * 255))
        print(img8bits.shape)
        print(self.roi_template.shape)
        res = cv.matchTemplate(img8bits, self.roi_template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        xy = peak_local_max(res, min_distance=self.roi_template.shape[0], threshold_abs=threshold, exclude_border=False)
        w, h = self.roi_template.shape[::-1]
        for pt in xy:
            x, y = pt[1], pt[0]
            rect_item = self.scene.addRect(QRect(0, 0, w, h))
            rect_item.setPen(self.scene.pen)
            rect_item.setPos(x, y)


class ViewerScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.rect_item = None
        self.from_x = None
        self.from_y = None
        self.pen = QPen(Qt.GlobalColor.cyan)
        self.pen.setWidth(2)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Delete):
            for r in self.selectedItems():
                self.removeItem(r)
            self.rect_item = None

    def mousePressEvent(self, event):
        if event.buttonDownScenePos(Qt.LeftButton):
            match PyDetecDiv().current_drawing_tool:
                case DrawingTools.Cursor:
                    self.select_ROI(event)
                case DrawingTools.DrawROI:
                    self.start_drawing_ROI(event)
                case DrawingTools.DuplicateROI:
                    self.duplicate_selected_ROI(event)

    def select_ROI(self, event):
        [r.setSelected(False) for r in self.items()]
        self.rect_item = None
        r = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
        if isinstance(r, QGraphicsRectItem):
            r.setSelected(True)
            self.rect_item = r
        if self.selectedItems():
            self.parent().ui.actionSet_template.setEnabled(True)
        else:
            self.parent().ui.actionSet_template.setEnabled(False)

    def start_drawing_ROI(self, event):
        pos = event.scenePos()
        self.rect_item = self.addRect(QRect(0, 0, 1, 1))
        self.rect_item.setPen(self.pen)
        self.rect_item.setPos(QPoint(pos.x(), pos.y()))
        self.rect_item.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.rect_item.setData(0, f'Rectangle{len(self.items())}')
        self.select_ROI(event)

    def duplicate_selected_ROI(self, event):
        pos = event.scenePos()
        if self.rect_item:
            self.rect_item = self.addRect(self.rect_item.rect())
            self.rect_item.setPen(self.pen)
            self.rect_item.setPos(
                QPoint(pos.x() - self.rect_item.rect().width() / 2.0, pos.y() - self.rect_item.rect().height() / 2.0))
            self.rect_item.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
            self.rect_item.setData(0, f'Rectangle{len(self.items())}')
            self.select_ROI(event)

    def mouseMoveEvent(self, event):
        if event.buttonDownScenePos(Qt.LeftButton):
            match PyDetecDiv().current_drawing_tool, event.modifiers():
                case DrawingTools.Cursor, Qt.NoModifier:
                    self.move_ROI(event)
                case DrawingTools.Cursor, Qt.ControlModifier:
                    self.draw_ROI(event)
                case DrawingTools.DrawROI, Qt.NoModifier:
                    self.draw_ROI(event)
                case DrawingTools.DuplicateROI, Qt.NoModifier:
                    self.move_ROI(event)

    def move_ROI(self, event):
        if self.rect_item:
            pos = event.scenePos()
            self.rect_item.moveBy(pos.x() - event.lastScenePos().x(), pos.y() - event.lastScenePos().y())

    def draw_ROI(self, event):
        if self.rect_item:
            pos = event.scenePos()
            roi_pos = self.rect_item.scenePos()
            rect = QRect(0, 0, pos.x() - roi_pos.x(), pos.y() - roi_pos.y())
            self.rect_item.setRect(rect)

    def wheelEvent(self, event):
        # event.ignore()
        ...
