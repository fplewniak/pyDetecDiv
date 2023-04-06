# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ImageViewer.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsView, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QSizePolicy,
    QSlider, QToolButton, QVBoxLayout, QWidget)

class Ui_ImageViewer(object):
    def setupUi(self, ImageViewer):
        if not ImageViewer.objectName():
            ImageViewer.setObjectName(u"ImageViewer")
        ImageViewer.resize(946, 656)
        ImageViewer.setWindowTitle(u"MainWindow")
        ImageViewer.setDockOptions(QMainWindow.AllowTabbedDocks|QMainWindow.AnimatedDocks|QMainWindow.ForceTabbedDocks)
        self.centralwidget = QWidget(ImageViewer)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.viewer = QGraphicsView(self.centralwidget)
        self.viewer.setObjectName(u"viewer")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.viewer.sizePolicy().hasHeightForWidth())
        self.viewer.setSizePolicy(sizePolicy)
        self.viewer.setFrameShape(QFrame.Box)
        self.viewer.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.viewer)

        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Box)
        self.frame.setFrameShadow(QFrame.Sunken)
        self.verticalLayout_2 = QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.controls = QGroupBox(self.frame)
        self.controls.setObjectName(u"controls")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.controls.sizePolicy().hasHeightForWidth())
        self.controls.setSizePolicy(sizePolicy1)
        self.controls.setFlat(False)
        self.controls.setCheckable(False)
        self.horizontalLayout_2 = QHBoxLayout(self.controls)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.video_back = QToolButton(self.controls)
        self.video_back.setObjectName(u"video_back")
        self.video_back.setText(u"...")
        icon = QIcon()
        icon.addFile(u":/icons/video_back", QSize(), QIcon.Normal, QIcon.Off)
        self.video_back.setIcon(icon)

        self.horizontalLayout_2.addWidget(self.video_back)

        self.toolButton = QToolButton(self.controls)
        self.toolButton.setObjectName(u"toolButton")
        self.toolButton.setText(u"...")
        icon1 = QIcon()
        icon1.addFile(u":/icons/video_pause", QSize(), QIcon.Normal, QIcon.Off)
        self.toolButton.setIcon(icon1)

        self.horizontalLayout_2.addWidget(self.toolButton)

        self.video_forward = QToolButton(self.controls)
        self.video_forward.setObjectName(u"video_forward")
        self.video_forward.setText(u"...")
        icon2 = QIcon()
        icon2.addFile(u":/icons/video_forward", QSize(), QIcon.Normal, QIcon.Off)
        self.video_forward.setIcon(icon2)

        self.horizontalLayout_2.addWidget(self.video_forward)

        self.T = QLabel(self.controls)
        self.T.setObjectName(u"T")
        self.T.setText(u"T")

        self.horizontalLayout_2.addWidget(self.T)

        self.t_slider = QSlider(self.controls)
        self.t_slider.setObjectName(u"t_slider")
        self.t_slider.setOrientation(Qt.Horizontal)
        self.t_slider.setTickPosition(QSlider.NoTicks)
        self.t_slider.setTickInterval(10)

        self.horizontalLayout_2.addWidget(self.t_slider)

        self.sep1 = QFrame(self.controls)
        self.sep1.setObjectName(u"sep1")
        self.sep1.setFrameShape(QFrame.VLine)
        self.sep1.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_2.addWidget(self.sep1)

        self.Z = QLabel(self.controls)
        self.Z.setObjectName(u"Z")
        self.Z.setText(u"Z")

        self.horizontalLayout_2.addWidget(self.Z)

        self.z_slider = QSlider(self.controls)
        self.z_slider.setObjectName(u"z_slider")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.z_slider.sizePolicy().hasHeightForWidth())
        self.z_slider.setSizePolicy(sizePolicy2)
        self.z_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_2.addWidget(self.z_slider)

        self.sep2 = QFrame(self.controls)
        self.sep2.setObjectName(u"sep2")
        self.sep2.setFrameShape(QFrame.VLine)
        self.sep2.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_2.addWidget(self.sep2)

        self.zoom_actual = QToolButton(self.controls)
        self.zoom_actual.setObjectName(u"zoom_actual")
        self.zoom_actual.setText(u"...")
        icon3 = QIcon()
        icon3.addFile(u":/icons/zoom_actual", QSize(), QIcon.Normal, QIcon.Off)
        self.zoom_actual.setIcon(icon3)

        self.horizontalLayout_2.addWidget(self.zoom_actual)

        self.zoom_fit = QToolButton(self.controls)
        self.zoom_fit.setObjectName(u"zoom_fit")
        self.zoom_fit.setText(u"...")
        icon4 = QIcon()
        icon4.addFile(u":/icons/zoom_fit", QSize(), QIcon.Normal, QIcon.Off)
        self.zoom_fit.setIcon(icon4)

        self.horizontalLayout_2.addWidget(self.zoom_fit)

        self.zoom_value = QSlider(self.controls)
        self.zoom_value.setObjectName(u"zoom_value")
        sizePolicy2.setHeightForWidth(self.zoom_value.sizePolicy().hasHeightForWidth())
        self.zoom_value.setSizePolicy(sizePolicy2)
        self.zoom_value.setMinimum(10)
        self.zoom_value.setMaximum(200)
        self.zoom_value.setValue(100)
        self.zoom_value.setOrientation(Qt.Horizontal)

        self.horizontalLayout_2.addWidget(self.zoom_value)


        self.verticalLayout_2.addWidget(self.controls)

        self.FOV_info = QGroupBox(self.frame)
        self.FOV_info.setObjectName(u"FOV_info")
        self.FOV_info.setTitle(u"")
        self.horizontalLayout = QHBoxLayout(self.FOV_info)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.view_name = QLabel(self.FOV_info)
        self.view_name.setObjectName(u"view_name")
        self.view_name.setText(u"View:")

        self.horizontalLayout.addWidget(self.view_name)

        self.current_frame = QLabel(self.FOV_info)
        self.current_frame.setObjectName(u"current_frame")
        self.current_frame.setText(u"Frame: 0")

        self.horizontalLayout.addWidget(self.current_frame)

        self.FPS = QLabel(self.FOV_info)
        self.FPS.setObjectName(u"FPS")
        self.FPS.setText(u"FPS: 0")

        self.horizontalLayout.addWidget(self.FPS)

        self.scale_value = QLabel(self.FOV_info)
        self.scale_value.setObjectName(u"scale_value")
        self.scale_value.setText(u"Zoom: 100%")

        self.horizontalLayout.addWidget(self.scale_value)


        self.verticalLayout_2.addWidget(self.FOV_info)


        self.verticalLayout.addWidget(self.frame)

        ImageViewer.setCentralWidget(self.centralwidget)

        self.retranslateUi(ImageViewer)
        self.video_back.clicked.connect(ImageViewer.video_back)
        self.zoom_actual.clicked.connect(ImageViewer.zoom_reset)
        self.toolButton.clicked.connect(ImageViewer.pause_video)
        self.t_slider.valueChanged.connect(ImageViewer.change_frame)
        self.z_slider.valueChanged.connect(ImageViewer.change_layer)
        self.zoom_value.sliderMoved.connect(ImageViewer.zoom_set_value)
        self.zoom_fit.clicked.connect(ImageViewer.zoom_fit)
        self.video_forward.clicked.connect(ImageViewer.play_video)

        QMetaObject.connectSlotsByName(ImageViewer)
    # setupUi

    def retranslateUi(self, ImageViewer):
        pass
    # retranslateUi

