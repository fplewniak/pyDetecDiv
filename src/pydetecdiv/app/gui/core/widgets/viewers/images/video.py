import time

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame

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

        layout = QVBoxLayout(self)
        self.viewer = ImageViewer()
        buttons = QFrame(self.viewer)
        button_layout = QHBoxLayout(buttons)
        buttons.setLayout(button_layout)
        button_up = QPushButton('Up', self.viewer)
        button_down = QPushButton('Down', self.viewer)
        button_start = QPushButton('Start', self.viewer)
        button_layout.addWidget(button_up)
        button_layout.addWidget(button_down)
        button_layout.addWidget(button_start)
        button_start.clicked.connect(self.play_video)
        layout.addWidget(self.viewer)
        layout.addWidget(buttons)
        self.setLayout(layout)

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
        frame = self.T + 1 #self.ui.t_step.value()
        if frame >= self.viewer.background.image.image_resource_data.sizeT or not self.video_playing:
            print('Stop video')
            self.timer.stop()
            print(self.speed)
        else:
            end = time.time()
            # speed = np.around((self.frame - self.first_frame) / ((end - self.start) * self.ui.t_step.value()), 1)
            self.change_frame(frame)
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


