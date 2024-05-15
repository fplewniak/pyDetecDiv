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
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsView, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMenu,
    QMenuBar, QSizePolicy, QSlider, QSpinBox,
    QToolButton, QVBoxLayout, QWidget)

class Ui_ImageViewer(object):
    def setupUi(self, ImageViewer):
        if not ImageViewer.objectName():
            ImageViewer.setObjectName(u"ImageViewer")
        ImageViewer.resize(946, 656)
        ImageViewer.setWindowTitle(u"MainWindow")
        ImageViewer.setDockOptions(QMainWindow.AllowTabbedDocks|QMainWindow.AnimatedDocks|QMainWindow.ForceTabbedDocks)
        self.actionApply_correction = QAction(ImageViewer)
        self.actionApply_correction.setObjectName(u"actionApply_correction")
        self.actionApply_correction.setCheckable(True)
        self.actionApply_correction.setEnabled(False)
        self.actionApply_correction.setText(u"Apply correction")
        self.actionClose_window = QAction(ImageViewer)
        self.actionClose_window.setObjectName(u"actionClose_window")
        self.actionClose_window.setText(u"Close window")
        self.actionPlot = QAction(ImageViewer)
        self.actionPlot.setObjectName(u"actionPlot")
        self.actionPlot.setEnabled(False)
        self.actionPlot.setText(u"Plot")
        self.actionSet_template = QAction(ImageViewer)
        self.actionSet_template.setObjectName(u"actionSet_template")
        self.actionSet_template.setEnabled(False)
        self.actionSet_template.setText(u"Selection as template")
        self.actionLoad_template = QAction(ImageViewer)
        self.actionLoad_template.setObjectName(u"actionLoad_template")
        self.actionLoad_template.setEnabled(True)
        self.actionLoad_template.setText(u"Load template")
        self.actionIdentify_ROIs = QAction(ImageViewer)
        self.actionIdentify_ROIs.setObjectName(u"actionIdentify_ROIs")
        self.actionIdentify_ROIs.setEnabled(False)
        self.actionIdentify_ROIs.setText(u"Detect ROIs")
        self.actionSave_ROIs = QAction(ImageViewer)
        self.actionSave_ROIs.setObjectName(u"actionSave_ROIs")
        self.actionSave_ROIs.setText(u"Save ROIs")
        self.actionView_template = QAction(ImageViewer)
        self.actionView_template.setObjectName(u"actionView_template")
        self.actionView_template.setEnabled(False)
        self.actionView_template.setText(u"View template in new tab")
        self.actionVidstab = QAction(ImageViewer)
        self.actionVidstab.setObjectName(u"actionVidstab")
        self.actionVidstab.setText(u"Vidstab")
        self.actionPhase_correlation = QAction(ImageViewer)
        self.actionPhase_correlation.setObjectName(u"actionPhase_correlation")
        self.actionPhase_correlation.setText(u"Phase correlation")
        self.actionLoad_file = QAction(ImageViewer)
        self.actionLoad_file.setObjectName(u"actionLoad_file")
        self.actionLoad_file.setText(u"Load file")
        self.actionSave_to_file = QAction(ImageViewer)
        self.actionSave_to_file.setObjectName(u"actionSave_to_file")
        self.actionSave_to_file.setEnabled(False)
        self.actionSave_to_file.setText(u"Save to file")
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
        self.video_back_btn = QToolButton(self.controls)
        self.video_back_btn.setObjectName(u"video_back_btn")
        self.video_back_btn.setText(u"...")
        icon = QIcon()
        icon.addFile(u":/icons/video_back", QSize(), QIcon.Normal, QIcon.Off)
        self.video_back_btn.setIcon(icon)

        self.horizontalLayout_2.addWidget(self.video_back_btn)

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
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.t_slider.sizePolicy().hasHeightForWidth())
        self.t_slider.setSizePolicy(sizePolicy2)
        self.t_slider.setOrientation(Qt.Horizontal)
        self.t_slider.setTickPosition(QSlider.NoTicks)
        self.t_slider.setTickInterval(10)

        self.horizontalLayout_2.addWidget(self.t_slider)

        self.label = QLabel(self.controls)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.t_step = QSpinBox(self.controls)
        self.t_step.setObjectName(u"t_step")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.t_step.sizePolicy().hasHeightForWidth())
        self.t_step.setSizePolicy(sizePolicy3)
        self.t_step.setMinimum(1)
        self.t_step.setMaximum(999)

        self.horizontalLayout_2.addWidget(self.t_step)

        self.sep3 = QFrame(self.controls)
        self.sep3.setObjectName(u"sep3")
        self.sep3.setFrameShape(QFrame.VLine)
        self.sep3.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_2.addWidget(self.sep3)

        self.C = QLabel(self.controls)
        self.C.setObjectName(u"C")

        self.horizontalLayout_2.addWidget(self.C)

        self.c_slider = QSlider(self.controls)
        self.c_slider.setObjectName(u"c_slider")
        sizePolicy3.setHeightForWidth(self.c_slider.sizePolicy().hasHeightForWidth())
        self.c_slider.setSizePolicy(sizePolicy3)
        self.c_slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_2.addWidget(self.c_slider)

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
        sizePolicy3.setHeightForWidth(self.z_slider.sizePolicy().hasHeightForWidth())
        self.z_slider.setSizePolicy(sizePolicy3)
        self.z_slider.setPageStep(1)
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

        self.zoom_fit_btn = QToolButton(self.controls)
        self.zoom_fit_btn.setObjectName(u"zoom_fit_btn")
        self.zoom_fit_btn.setText(u"...")
        icon4 = QIcon()
        icon4.addFile(u":/icons/zoom_fit", QSize(), QIcon.Normal, QIcon.Off)
        self.zoom_fit_btn.setIcon(icon4)

        self.horizontalLayout_2.addWidget(self.zoom_fit_btn)

        self.zoom_value = QSlider(self.controls)
        self.zoom_value.setObjectName(u"zoom_value")
        sizePolicy3.setHeightForWidth(self.zoom_value.sizePolicy().hasHeightForWidth())
        self.zoom_value.setSizePolicy(sizePolicy3)
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
        sizePolicy3.setHeightForWidth(self.FPS.sizePolicy().hasHeightForWidth())
        self.FPS.setSizePolicy(sizePolicy3)
        self.FPS.setText(u"FPS: 0")

        self.horizontalLayout.addWidget(self.FPS)

        self.scale_value = QLabel(self.FOV_info)
        self.scale_value.setObjectName(u"scale_value")
        self.scale_value.setText(u"Zoom: 100%")

        self.horizontalLayout.addWidget(self.scale_value)


        self.verticalLayout_2.addWidget(self.FOV_info)


        self.verticalLayout.addWidget(self.frame)

        ImageViewer.setCentralWidget(self.centralwidget)
        self.menuBar = QMenuBar(ImageViewer)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 946, 19))
        self.menuResource = QMenu(self.menuBar)
        self.menuResource.setObjectName(u"menuResource")
        self.menuResource.setTitle(u"Resource")
        self.menuDrift = QMenu(self.menuResource)
        self.menuDrift.setObjectName(u"menuDrift")
        self.menuDrift.setTitle(u"Drift")
        self.menuCompute_and_plot = QMenu(self.menuDrift)
        self.menuCompute_and_plot.setObjectName(u"menuCompute_and_plot")
        self.menuCompute_and_plot.setTitle(u"Compute and plot")
        self.menuROI = QMenu(self.menuBar)
        self.menuROI.setObjectName(u"menuROI")
        self.menuROI.setTitle(u"ROI")
        ImageViewer.setMenuBar(self.menuBar)

        self.menuBar.addAction(self.menuResource.menuAction())
        self.menuBar.addAction(self.menuROI.menuAction())
        self.menuResource.addAction(self.menuDrift.menuAction())
        self.menuResource.addSeparator()
        self.menuDrift.addAction(self.menuCompute_and_plot.menuAction())
        self.menuDrift.addAction(self.actionLoad_file)
        self.menuDrift.addSeparator()
        self.menuDrift.addAction(self.actionSave_to_file)
        self.menuDrift.addAction(self.actionPlot)
        self.menuDrift.addSeparator()
        self.menuDrift.addAction(self.actionApply_correction)
        self.menuCompute_and_plot.addAction(self.actionVidstab)
        self.menuCompute_and_plot.addAction(self.actionPhase_correlation)
        self.menuROI.addSeparator()
        self.menuROI.addAction(self.actionSet_template)
        self.menuROI.addAction(self.actionLoad_template)
        self.menuROI.addSeparator()
        self.menuROI.addAction(self.actionView_template)
        self.menuROI.addAction(self.actionIdentify_ROIs)
        self.menuROI.addAction(self.actionSave_ROIs)

        self.retranslateUi(ImageViewer)
        self.video_back_btn.clicked.connect(ImageViewer.video_back)
        self.zoom_actual.clicked.connect(ImageViewer.zoom_reset)
        self.toolButton.clicked.connect(ImageViewer.pause_video)
        self.t_slider.valueChanged.connect(ImageViewer.change_frame)
        self.z_slider.valueChanged.connect(ImageViewer.change_layer)
        self.zoom_value.sliderMoved.connect(ImageViewer.zoom_set_value)
        self.zoom_fit_btn.clicked.connect(ImageViewer.zoom_fit)
        self.video_forward.clicked.connect(ImageViewer.play_video)
        self.actionClose_window.triggered.connect(ImageViewer.close_window)
        self.actionApply_correction.triggered.connect(ImageViewer.apply_drift_correction)
        # self.actionPlot.triggered.connect(ImageViewer.plot_drift)
        self.actionSet_template.triggered.connect(ImageViewer.set_roi_template)
        self.actionLoad_template.triggered.connect(ImageViewer.load_roi_template)
        self.actionIdentify_ROIs.triggered.connect(ImageViewer.identify_rois)
        self.actionView_template.triggered.connect(ImageViewer.view_template)
        self.actionSave_ROIs.triggered.connect(ImageViewer.save_rois)
        # self.actionVidstab.triggered.connect(ImageViewer.compute_drift_vidstab)
        # self.actionPhase_correlation.triggered.connect(ImageViewer.compute_drift_phase_correlation)
        # self.actionSave_to_file.triggered.connect(ImageViewer.save_drift_file)
        # self.actionLoad_file.triggered.connect(ImageViewer.load_drift_file)
        self.c_slider.valueChanged.connect(ImageViewer.change_channel)

        QMetaObject.connectSlotsByName(ImageViewer)
    # setupUi

    def retranslateUi(self, ImageViewer):
        self.label.setText(QCoreApplication.translate("ImageViewer", u"step", None))
        self.C.setText(QCoreApplication.translate("ImageViewer", u"C", None))
        pass
    # retranslateUi

