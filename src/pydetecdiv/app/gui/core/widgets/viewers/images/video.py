"""
Module defining classes for building a video player
"""
import time
import re
from typing import TypeVar

import numpy as np
from PySide6.QtCore import QTimer, Signal, QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame, QToolButton, QLabel, QSlider, QSpinBox,
                               QLineEdit, QSizePolicy, QSplitter, QMenuBar, QGraphicsScene)

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers import Layer
from pydetecdiv.app.gui.core.widgets.viewers.images import ImageViewer
from pydetecdiv.domain.ImageResourceData import ImageResourceData

VideoScene = TypeVar('VideoScene', bound=QGraphicsScene)


class VideoPlayer(QWidget):
    """
    A class defining a video player
    """
    video_frame = Signal(int)
    video_channel = Signal(int)
    video_Z = Signal(int)
    finished = Signal(bool)

    def __init__(self, parent: QWidget = None, **kwargs):
        super().__init__(parent, **kwargs)

        self.T = 0
        self.start = 0
        self.timer = None
        self.speed = 0
        self.first_frame = self.T
        self.video_playing = False
        self.viewer_panel = None
        self.control_panel = None
        self.menubar = None
        self.tscale = 1
        PyDetecDiv.main_window.active_subwindow.tabBarClicked.connect(self.other_scene_in_focus)

        # self.viewer.setViewportUpdateMode(QGraphicsView.NoViewportUpdate)

    def other_scene_in_focus(self, tab):
        ...
    @property
    def elapsed_time(self) -> str:
        t = self.T * self.tscale
        hours = int(t / 3600)
        t = t - hours * 3600
        minutes = int(t / 60)
        t = t - minutes * 60
        seconds = int(t)
        ms = int(1000 * (t - seconds))
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:04d}'

    @property
    def viewer(self) -> ImageViewer:
        return self.viewer_panel.video_viewer

    # def _create_viewer(self):
    #     """
    #     Create the viewer for this video player
    #
    #     :return: ImageViewer, the created viewer
    #     """
    #     # viewer = ImageViewer()
    #     # viewer.setup()
    #     self.viewer_panel = VideoViewerPanel(self)
    #     return self.viewer_panel.video_viewer

    def setup(self, menubar: QMenuBar = None) -> None:
        """
        Sets the video player up

        :param menubar: whether a menubar should be added or not
        """
        layout = QVBoxLayout(self)
        if menubar:
            self.menubar = menubar
            layout.addWidget(self.menubar)
        # self._create_viewer()
        self.viewer_panel = VideoViewerPanel(self)
        self.control_panel = VideoControlPanel(self)
        layout.addWidget(self.viewer_panel)
        layout.addWidget(self.control_panel)
        self.setLayout(layout)

        self.time_display = QLabel(self.elapsed_time, parent=self)
        self.time_display.setStyleSheet("color: green; font-size: 18px;")
        self.time_display.setGeometry(20, 30, 140, self.time_display.height())

    @property
    def scene(self) -> VideoScene:
        """
        Convenience property returning the scene associated with the viewer in this video player

        :return: the video player scene
        """
        return self.viewer.scene()

    def setBackgroundImage(self, image_resource_data: ImageResourceData, C: int = 0, Z: int = 0, T: int = 0,
                           crop: tuple[slice, slice] = None) -> None:
        """
        Sets the background image for this video player

        :param image_resource_data: the Image resource data
        :param C: the channel of tuple of channels
        :param Z: the z-slice
        :param T: the frame index
        :param crop: the crop values
        """
        self.viewer.setBackgroundImage(image_resource_data, C=C, Z=Z, T=T, crop=crop)
        self.T = T
        self.time_display.setText(self.elapsed_time)

        self.control_panel.video_control.t_slider.setMinimum(0)
        self.control_panel.video_control.t_slider.setMaximum(image_resource_data.sizeT - 1)
        self.control_panel.video_control.t_slider.setEnabled(True)
        self.control_panel.video_control.t_slider.setValue(self.T)
        self.control_panel.video_control.t_set.setText(str(self.T))
        self.control_panel.video_control.t_step.setMaximum(image_resource_data.sizeT - 1)

        # timeLine = QTimeLine(1000*image_resource_data.sizeT/20, self)
        # timeLine.setFrameRange(0, image_resource_data.sizeT - 1)
        # timeLine.frameChanged.connect(self.change_frame)
        # # Clicking the push button will start the progress bar animation
        # self.control_panel.video_control.video_forward.clicked.connect(timeLine.start)

    def addLayer(self, name=None) -> Layer:
        """
        add a layer to the scene of the current Video player

        :return: ImageLayer, the new layer
        """
        return self.viewer.addLayer(name=name)

    def zoom_set_value(self, value: int) -> None:
        """
        Set the zoom to the specified value

        :param value: the zoom value (as %)
        :type value: float
        """
        self.viewer.zoom_set_value(value)
        self.control_panel.zoom_control.zoom_set.setText(f'{self.viewer.scale_value} %')
        # self.control_panel.zoom_control.zoom_value.setValue(value)
        # print(value)
        # self.scale_value.setText(f'Zoom: {self.scale}%')

    def zoom_reset(self) -> None:
        """
        Reset the zoom to 1:1
        """
        self.viewer.zoom_set_value(100)
        # self.control_panel.zoom_control.zoom_value.setValue(100)
        self.control_panel.zoom_control.zoom_set.setText('100 %')
        # self.zoom_set_value(100)

    def zoom_fit(self) -> None:
        """
        Set the zoom value to fit the image in the viewer
        """
        self.viewer.zoom_fit()
        # self.control_panel.zoom_control.zoom_value.setSliderPosition(self.viewer.scale_value)
        self.control_panel.zoom_control.zoom_set.setText(f'{self.viewer.scale_value} %')
        # self.control_panel.zoom_control.scale_value.setText(f'Zoom: {self.scale}%')

    def play_video(self) -> None:
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

    def show_next_frame(self) -> None:
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

    def change_frame(self, T: int = 0) -> None:
        """
        Change the current frame to the specified time index and refresh the display

        :param T:
        """
        if self.T != T:
            self.T = T
            self.time_display.setText(self.elapsed_time)
            self.video_frame.emit(self.T)
            self.control_panel.video_control.t_slider.setValue(self.T)
            self.control_panel.video_control.t_set.setText(str(self.T))
            self.viewer.display(T=self.T)
            # self.viewer.scene().update()

    def pause_video(self) -> None:
        """
        Pause the video by setting the video_playing flag to False
        """
        self.video_playing = False

    def video_back(self) -> None:
        """
        Reset the current frame to the first one
        """
        self.change_frame(T=0)
        self.control_panel.video_control.t_slider.setValue(0)

    def synchronize_with(self, other: 'VideoPlayer') -> None:
        """
        Synchronize the current viewer with another one. This is used to view a portion of an image in a new viewer
        with the same T coordinates as the original.

        :param other: the other viewer to synchronize with
        """
        self.change_frame(other.T)
        # if self.T != other.T:
        #     self.T = other.T
        #     self.video_frame.emit(self.T)
        #     self.control_panel.video_control.t_slider.setValue(self.T)
        #     self.control_panel.video_control.t_set.setText(str(self.T))
        # self.viewer.display(T=self.T)


class VideoViewerPanel(QSplitter):
    """
    A class defining the Video viewer panel (holding image viewer + any other viewer for graphs, etc...)
    """

    def __init__(self, parent: VideoPlayer = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.video_viewer = ImageViewer()

    def setup(self, scene: VideoScene = None, **kwargs) -> None:
        self.video_viewer.setup(scene)
        layout = QVBoxLayout(self)
        layout.addWidget(self.video_viewer)
        self.setLayout(layout)


class VideoControlPanel(QFrame):
    """
    A class defining the Video control panel (video control + zoom control)
    """

    def __init__(self, parent: VideoPlayer = None, **kwargs):
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
        self.video_control.t_set.returnPressed.connect(lambda: parent.change_frame(T=int(self.video_control.frame)))
        # self.zoom_control.zoom_value.valueChanged.connect(parent.zoom_set_value)
        self.zoom_control.zoom_fit_btn.clicked.connect(parent.zoom_fit)
        self.zoom_control.zoom_actual.clicked.connect(parent.zoom_reset)
        self.zoom_control.zoom_set.returnPressed.connect(lambda: parent.zoom_set_value(int(self.zoom_control.zoom)))
        parent.viewer.zoom_value_changed.connect(lambda x: self.zoom_control.zoom_set.setText(f'{x} %'))

    def addWidget(self, widget: QWidget) -> None:
        self.layout().addWidget(widget)


class VideoControl(QFrame):
    """
    A class defining the frame Controls (frame, frame rate, play, pause, etc.) of a VideoPlayer
    """

    def __init__(self, parent: VideoControlPanel = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        layout = QHBoxLayout(self)
        self.video_back_btn = QToolButton(self)
        icon = QIcon()
        icon.addFile(':/icons/video_back', QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.video_back_btn.setIcon(icon)

        layout.addWidget(self.video_back_btn)

        self.pauseButton = QToolButton(self)
        icon1 = QIcon()
        icon1.addFile(':/icons/video_pause', QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pauseButton.setIcon(icon1)

        layout.addWidget(self.pauseButton)

        self.video_forward = QToolButton(self)
        icon2 = QIcon()
        icon2.addFile(':/icons/video_play', QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.video_forward.setIcon(icon2)

        layout.addWidget(self.video_forward)

        self.t_set = QLineEdit(self)
        self.t_set.setText(str(self.parent().parent().T))
        self.t_set.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.t_set.setMaxLength(5)
        self.t_set.setMaximumWidth(40)
        self.t_set.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        layout.addWidget(self.t_set)

        self.t_slider = QSlider(self)
        self.t_slider.setOrientation(Qt.Orientation.Horizontal)
        self.t_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.t_slider.setTickInterval(10)

        layout.addWidget(self.t_slider)

        layout.addWidget(QLabel('step:', self))

        self.t_step = QSpinBox(self)
        self.t_step.setMinimum(1)

        layout.addWidget(self.t_step)

    @property
    def frame(self) -> int:
        """
        Get the current frame displayed by the Video Player
        :return: the current frame number
        """
        return int(self.t_set.text())


class ZoomControl(QFrame):
    """
    A class defining a QFrame widget for controlling Zoom in a VideoPlayer
    """

    def __init__(self, parent: VideoControlPanel = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        layout = QHBoxLayout(self)

        self.zoom_actual = QToolButton(self)
        icon3 = QIcon()
        icon3.addFile(':/icons/zoom_actual', QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.zoom_actual.setIcon(icon3)

        layout.addWidget(self.zoom_actual)

        self.zoom_fit_btn = QToolButton(self)
        icon4 = QIcon()
        icon4.addFile(":/icons/zoom_fit", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.zoom_fit_btn.setIcon(icon4)

        layout.addWidget(self.zoom_fit_btn)

        self.zoom_set = QLineEdit(self)
        self.zoom_set.setText('100 %')
        self.zoom_set.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.zoom_set.setMaxLength(6)
        self.zoom_set.setMaximumWidth(45)
        self.zoom_set.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        layout.addWidget(self.zoom_set)

        # self.zoom_value = QSlider(self)
        # # sizePolicy3.setHeightForWidth(self.zoom_value.sizePolicy().hasHeightForWidth())
        # # self.zoom_value.setSizePolicy(sizePolicy3)
        # self.zoom_value.setMinimum(10)
        # self.zoom_value.setMaximum(1000)
        # self.zoom_value.setValue(100)
        # self.zoom_value.setOrientation(Qt.Horizontal)
        #
        # layout.addWidget(self.zoom_value)

    @property
    def zoom(self) -> int:
        """
        Get the current zoom level
        :return: the current zoom level
        """
        return int(re.match(r'\d+', self.zoom_set.text()).group(0))
