from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QMainWindow, QGraphicsScene, QApplication
import time
import numpy as np

from pydetecdiv.app.gui.ui.ImageViewer import Ui_ImageViewer


class ImageViewer(QMainWindow, Ui_ImageViewer):
    video_frame = Signal(int)

    def __init__(self, **kwargs):
        QMainWindow.__init__(self)
        self.ui = Ui_ImageViewer()
        self.setWindowTitle('Image viewer')
        self.ui.setupUi(self)
        self.scale = 100
        self.ui.z_slider.setEnabled(False)
        self.ui.t_slider.setEnabled(False)
        self.image_resource = None
        self.scene = QGraphicsScene()
        self.pixmap = QPixmap()
        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.ui.viewer.setScene(self.scene)
        self.C = 0
        self.T = 0
        self.Z = 0
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
        self.display()

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
