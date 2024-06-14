import time

import numpy as np
from PySide6.QtCore import QTimer, Signal, QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QToolButton, QLabel, QSlider, QSpinBox

from pydetecdiv.app.gui.core.widgets.viewers.images import ImageViewer


class VideoPlayer(QWidget):
    video_frame = Signal(int)
    video_channel = Signal(int)
    video_Z = Signal(int)
    finished = Signal(bool)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.T = 0
        self.start = 0
        self.first_frame = self.T
        self.video_playing = False
        self.viewer = None
        self.control_panel = None
        self.menubar = None

        # self.viewer.setViewportUpdateMode(QGraphicsView.NoViewportUpdate)

    def _create_viewer(self):
        viewer = ImageViewer()
        viewer.setup()
        return viewer

    def setup(self, menubar=None):
        layout = QVBoxLayout(self)
        if menubar:
            self.menubar = menubar
            layout.addWidget(self.menubar)
        self.viewer = self._create_viewer()
        self.control_panel = VideoControlPanel(self)
        layout.addWidget(self.viewer)
        layout.addWidget(self.control_panel)
        self.setLayout(layout)

    @property
    def scene(self):
        return self.viewer.scene()

    def setBackgroundImage(self, image_resource_data, C=0, Z=0, T=0, crop=None):
        self.viewer.setBackgroundImage(image_resource_data, C=C, Z=Z, T=T, crop=crop)
        self.T = T

        self.control_panel.video_control.t_slider.setMinimum(0)
        self.control_panel.video_control.t_slider.setMaximum(image_resource_data.sizeT - 1)
        self.control_panel.video_control.t_slider.setEnabled(True)
        self.control_panel.video_control.t_slider.setValue(self.T)
        self.control_panel.video_control.t_step.setMaximum(image_resource_data.sizeT - 1)

    def addLayer(self):
        return self.viewer.addLayer()

    def zoom_set_value(self, value):
        """
        Set the zoom to the specified value

        :param value: the zoom value (as %)
        :type value: float
        """
        self.viewer.zoom_set_value(value)
        # self.control_panel.zoom_control.zoom_value.setValue(value)
        # print(value)
        # self.scale_value.setText(f'Zoom: {self.scale}%')

    def zoom_reset(self):
        """
        Reset the zoom to 1:1
        """
        self.control_panel.zoom_control.zoom_value.setValue(100)
        # self.zoom_set_value(100)

    def zoom_fit(self):
        """
        Set the zoom value to fit the image in the viewer
        """
        self.viewer.zoom_fit()
        self.control_panel.zoom_control.zoom_value.setSliderPosition(self.viewer.scale_value)
        # self.control_panel.zoom_control.scale_value.setText(f'Zoom: {self.scale}%')

    def play_video(self):
        """
        Play the video with a maximum of an image every 50 ms (i.e. 20 FPS). Note that if loading a frame takes longer
        than 50 ms, then the frame rate may be lower.
        """
        print('Playing video')
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_next_frame)
        self.timer.setInterval(50)
        self.first_frame = self.T
        self.start = time.time()
        self.video_playing = True
        self.timer.start()

    def show_next_frame(self):
        """
        Show next frame when playing a video
        """
        frame = self.T + self.control_panel.video_control.t_step.value()
        if frame >= self.viewer.background.image.image_resource_data.sizeT or not self.video_playing:
            print('Stop video')
            self.timer.stop()
            print(self.speed)
        else:
            end = time.time()
            # speed = np.around((self.frame - self.first_frame) / ((end - self.start) * self.ui.t_step.value()), 1)
            self.change_frame(frame)
            self.control_panel.video_control.t_slider.setValue(self.T)
            speed = np.around((frame - self.first_frame) / (end - self.start), 1)
            self.speed = speed
            # self.ui.FPS.setText(f'FPS: {speed}')

    def change_frame(self, T=0):
        """
        Change the current frame to the specified time index and refresh the display

        :param T:
        """
        if self.T != T:
            self.T = T
            self.video_frame.emit(self.T)
            self.viewer.display(T=self.T)
            # self.viewer.scene().update()

    def pause_video(self):
        """
        Pause the video by setting the video_playing flag to False
        """
        self.video_playing = False

    def video_back(self):
        """
        Reset the current frame to the first one
        """
        self.change_frame(T=0)
        self.control_panel.video_control.t_slider.setValue(0)

class VideoControlPanel(QFrame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        layout = QHBoxLayout(self)

        self.video_control = VideoControl(self)
        layout.addWidget(self.video_control)

        self.zoom_control = ZoomControl(self)
        layout.addWidget(self.zoom_control)

        self.setLayout(layout)

        self.video_control.video_forward.clicked.connect(parent.play_video)
        self.video_control.pauseButton.clicked.connect(parent.pause_video)
        self.video_control.video_back_btn.clicked.connect(parent.video_back)
        self.video_control.t_slider.valueChanged.connect(parent.change_frame)
        self.zoom_control.zoom_value.valueChanged.connect(parent.zoom_set_value)
        self.zoom_control.zoom_fit_btn.clicked.connect(parent.zoom_fit)
        self.zoom_control.zoom_actual.clicked.connect(parent.zoom_reset)

class VideoControl(QFrame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        layout = QHBoxLayout(self)
        self.video_back_btn = QToolButton(self)
        icon = QIcon()
        icon.addFile(':/icons/video_back', QSize(), QIcon.Normal, QIcon.Off)
        self.video_back_btn.setIcon(icon)

        layout.addWidget(self.video_back_btn)

        self.pauseButton = QToolButton(self)
        icon1 = QIcon()
        icon1.addFile(u":/icons/video_pause", QSize(), QIcon.Normal, QIcon.Off)
        self.pauseButton.setIcon(icon1)

        layout.addWidget(self.pauseButton)

        self.video_forward = QToolButton(self)
        icon2 = QIcon()
        icon2.addFile(u":/icons/video_play", QSize(), QIcon.Normal, QIcon.Off)
        self.video_forward.setIcon(icon2)

        layout.addWidget(self.video_forward)

        self.T = QLabel(self)

        layout.addWidget(self.T)

        self.t_slider = QSlider(self)
        # sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # sizePolicy2.setHorizontalStretch(0)
        # sizePolicy2.setVerticalStretch(0)
        # sizePolicy2.setHeightForWidth(self.t_slider.sizePolicy().hasHeightForWidth())
        # self.t_slider.setSizePolicy(sizePolicy2)
        self.t_slider.setOrientation(Qt.Horizontal)
        self.t_slider.setTickPosition(QSlider.NoTicks)
        self.t_slider.setTickInterval(10)

        layout.addWidget(self.t_slider)

        layout.addWidget(QLabel('step', self))

        self.t_step = QSpinBox(self)
        # sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # sizePolicy3.setHorizontalStretch(0)
        # sizePolicy3.setVerticalStretch(0)
        # sizePolicy3.setHeightForWidth(self.t_step.sizePolicy().hasHeightForWidth())
        # self.t_step.setSizePolicy(sizePolicy3)
        self.t_step.setMinimum(1)

        layout.addWidget(self.t_step)

class ZoomControl(QFrame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        layout = QHBoxLayout(self)

        self.zoom_actual = QToolButton(self)
        icon3 = QIcon()
        icon3.addFile(':/icons/zoom_actual', QSize(), QIcon.Normal, QIcon.Off)
        self.zoom_actual.setIcon(icon3)

        layout.addWidget(self.zoom_actual)

        self.zoom_fit_btn = QToolButton(self)
        icon4 = QIcon()
        icon4.addFile(":/icons/zoom_fit", QSize(), QIcon.Normal, QIcon.Off)
        self.zoom_fit_btn.setIcon(icon4)

        layout.addWidget(self.zoom_fit_btn)

        self.zoom_value = QSlider(self)
        # sizePolicy3.setHeightForWidth(self.zoom_value.sizePolicy().hasHeightForWidth())
        # self.zoom_value.setSizePolicy(sizePolicy3)
        self.zoom_value.setMinimum(10)
        self.zoom_value.setMaximum(1000)
        self.zoom_value.setValue(100)
        self.zoom_value.setOrientation(Qt.Horizontal)

        layout.addWidget(self.zoom_value)
