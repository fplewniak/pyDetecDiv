# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ImportAnnotatedROIs.ui'
##
## Created by: Qt User Interface Compiler version 6.6.2
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QDialog,
    QDialogButtonBox, QFrame, QHBoxLayout, QLabel,
    QSizePolicy, QVBoxLayout, QWidget)

from pydetecdiv.app.gui.ui.QLabelClickable import QLabelClickable

class Ui_FOV2ROIlinks(object):
    def setupUi(self, FOV2ROIlinks):
        if not FOV2ROIlinks.objectName():
            FOV2ROIlinks.setObjectName(u"FOV2ROIlinks")
        FOV2ROIlinks.resize(532, 295)
        self.horizontalLayout = QHBoxLayout(FOV2ROIlinks)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.container = QWidget(FOV2ROIlinks)
        self.container.setObjectName(u"container")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.container.sizePolicy().hasHeightForWidth())
        self.container.setSizePolicy(sizePolicy)
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setObjectName(u"container_layout")
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.control_widgets = QWidget(self.container)
        self.control_widgets.setObjectName(u"control_widgets")
        sizePolicy.setHeightForWidth(self.control_widgets.sizePolicy().hasHeightForWidth())
        self.control_widgets.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(self.control_widgets)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(self.control_widgets)
        self.label.setObjectName(u"label")
        self.label.setStyleSheet(u"font: bold;")

        self.verticalLayout.addWidget(self.label)

        self.patterns_widget = QWidget(self.control_widgets)
        self.patterns_widget.setObjectName(u"patterns_widget")
        self.patterns_layout = QVBoxLayout(self.patterns_widget)
        self.patterns_layout.setObjectName(u"patterns_layout")
        self.pos_widget = QWidget(self.patterns_widget)
        self.pos_widget.setObjectName(u"pos_widget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pos_widget.sizePolicy().hasHeightForWidth())
        self.pos_widget.setSizePolicy(sizePolicy1)
        self.pos_layout = QHBoxLayout(self.pos_widget)
        self.pos_layout.setObjectName(u"pos_layout")
        self.pos_layout.setContentsMargins(0, 0, 0, 0)
        self.position = QLabel(self.pos_widget)
        self.position.setObjectName(u"position")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.position.sizePolicy().hasHeightForWidth())
        self.position.setSizePolicy(sizePolicy2)
        self.position.setMinimumSize(QSize(45, 0))
        self.position.setText(u"Position")

        self.pos_layout.addWidget(self.position)

        self.pos_left = QComboBox(self.pos_widget)
        self.pos_left.setObjectName(u"pos_left")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.pos_left.sizePolicy().hasHeightForWidth())
        self.pos_left.setSizePolicy(sizePolicy3)
        self.pos_left.setEditable(True)
        self.pos_left.setCurrentText(u"")

        self.pos_layout.addWidget(self.pos_left)

        self.pos_pattern = QComboBox(self.pos_widget)
        self.pos_pattern.setObjectName(u"pos_pattern")
        sizePolicy3.setHeightForWidth(self.pos_pattern.sizePolicy().hasHeightForWidth())
        self.pos_pattern.setSizePolicy(sizePolicy3)
        self.pos_pattern.setStyleSheet(u"border: 2px solid rgb(255, 127, 0);")
        self.pos_pattern.setEditable(True)
        self.pos_pattern.setCurrentText(u"")

        self.pos_layout.addWidget(self.pos_pattern)

        self.pos_right = QComboBox(self.pos_widget)
        self.pos_right.setObjectName(u"pos_right")
        sizePolicy3.setHeightForWidth(self.pos_right.sizePolicy().hasHeightForWidth())
        self.pos_right.setSizePolicy(sizePolicy3)
        self.pos_right.setEditable(True)

        self.pos_layout.addWidget(self.pos_right)

        self.pos_colour = QLabelClickable(self.pos_widget)
        self.pos_colour.setObjectName(u"pos_colour")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(20)
        sizePolicy4.setVerticalStretch(20)
        sizePolicy4.setHeightForWidth(self.pos_colour.sizePolicy().hasHeightForWidth())
        self.pos_colour.setSizePolicy(sizePolicy4)
        self.pos_colour.setMinimumSize(QSize(20, 20))
        self.pos_colour.setStyleSheet(u"background-color: rgb(255, 127, 0);")
        self.pos_colour.setFrameShape(QFrame.Panel)
        self.pos_colour.setFrameShadow(QFrame.Sunken)
        self.pos_colour.setLineWidth(3)

        self.pos_layout.addWidget(self.pos_colour)


        self.patterns_layout.addWidget(self.pos_widget)


        self.verticalLayout.addWidget(self.patterns_widget)


        self.container_layout.addWidget(self.control_widgets)

        self.samples_widget = QWidget(self.container)
        self.samples_widget.setObjectName(u"samples_widget")
        self.samples_layout_2 = QHBoxLayout(self.samples_widget)
        self.samples_layout_2.setObjectName(u"samples_layout_2")
        self.samples_layout_2.setContentsMargins(-1, -1, 9, -1)
        self.samplesROI_widget = QWidget(self.samples_widget)
        self.samplesROI_widget.setObjectName(u"samplesROI_widget")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.samplesROI_widget.sizePolicy().hasHeightForWidth())
        self.samplesROI_widget.setSizePolicy(sizePolicy5)
        self.samplesROI_layout = QVBoxLayout(self.samplesROI_widget)
        self.samplesROI_layout.setObjectName(u"samplesROI_layout")
        self.samplesROI_layout.setContentsMargins(-1, 0, -1, -1)
        self.samplesROI_box_title = QLabel(self.samplesROI_widget)
        self.samplesROI_box_title.setObjectName(u"samplesROI_box_title")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.samplesROI_box_title.sizePolicy().hasHeightForWidth())
        self.samplesROI_box_title.setSizePolicy(sizePolicy6)
        self.samplesROI_box_title.setStyleSheet(u"font: bold;")
        self.samplesROI_box_title.setText(u"Sample ROI names")
        self.samplesROI_box_title.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.samplesROI_layout.addWidget(self.samplesROI_box_title)

        self.sampleROI_list = QFrame(self.samplesROI_widget)
        self.sampleROI_list.setObjectName(u"sampleROI_list")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.sampleROI_list.sizePolicy().hasHeightForWidth())
        self.sampleROI_list.setSizePolicy(sizePolicy7)
        self.sampleROI_list.setFrameShape(QFrame.Box)
        self.sampleROI_list.setFrameShadow(QFrame.Sunken)
        self.sample_list_layout = QVBoxLayout(self.sampleROI_list)
        self.sample_list_layout.setObjectName(u"sample_list_layout")
        self.sampleROI1 = QLabel(self.sampleROI_list)
        self.sampleROI1.setObjectName(u"sampleROI1")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.sampleROI1.sizePolicy().hasHeightForWidth())
        self.sampleROI1.setSizePolicy(sizePolicy8)

        self.sample_list_layout.addWidget(self.sampleROI1)

        self.sampleROI2 = QLabel(self.sampleROI_list)
        self.sampleROI2.setObjectName(u"sampleROI2")
        sizePolicy8.setHeightForWidth(self.sampleROI2.sizePolicy().hasHeightForWidth())
        self.sampleROI2.setSizePolicy(sizePolicy8)

        self.sample_list_layout.addWidget(self.sampleROI2)

        self.sampleROI3 = QLabel(self.sampleROI_list)
        self.sampleROI3.setObjectName(u"sampleROI3")
        sizePolicy8.setHeightForWidth(self.sampleROI3.sizePolicy().hasHeightForWidth())
        self.sampleROI3.setSizePolicy(sizePolicy8)

        self.sample_list_layout.addWidget(self.sampleROI3)

        self.sampleROI4 = QLabel(self.sampleROI_list)
        self.sampleROI4.setObjectName(u"sampleROI4")
        sizePolicy8.setHeightForWidth(self.sampleROI4.sizePolicy().hasHeightForWidth())
        self.sampleROI4.setSizePolicy(sizePolicy8)

        self.sample_list_layout.addWidget(self.sampleROI4)

        self.sampleROI5 = QLabel(self.sampleROI_list)
        self.sampleROI5.setObjectName(u"sampleROI5")
        sizePolicy8.setHeightForWidth(self.sampleROI5.sizePolicy().hasHeightForWidth())
        self.sampleROI5.setSizePolicy(sizePolicy8)

        self.sample_list_layout.addWidget(self.sampleROI5)


        self.samplesROI_layout.addWidget(self.sampleROI_list)


        self.samples_layout_2.addWidget(self.samplesROI_widget)

        self.samplesFOV_widget = QWidget(self.samples_widget)
        self.samplesFOV_widget.setObjectName(u"samplesFOV_widget")
        self.sampleFOV_layout = QVBoxLayout(self.samplesFOV_widget)
        self.sampleFOV_layout.setObjectName(u"sampleFOV_layout")
        self.sampleFOV_layout.setContentsMargins(-1, 0, -1, -1)
        self.sampleFOV_box_title = QLabel(self.samplesFOV_widget)
        self.sampleFOV_box_title.setObjectName(u"sampleFOV_box_title")
        sizePolicy6.setHeightForWidth(self.sampleFOV_box_title.sizePolicy().hasHeightForWidth())
        self.sampleFOV_box_title.setSizePolicy(sizePolicy6)
        font = QFont()
        font.setBold(True)
        self.sampleFOV_box_title.setFont(font)
        self.sampleFOV_box_title.setText(u"Sample FOV names")

        self.sampleFOV_layout.addWidget(self.sampleFOV_box_title)

        self.sampleFOV_list = QFrame(self.samplesFOV_widget)
        self.sampleFOV_list.setObjectName(u"sampleFOV_list")
        sizePolicy7.setHeightForWidth(self.sampleFOV_list.sizePolicy().hasHeightForWidth())
        self.sampleFOV_list.setSizePolicy(sizePolicy7)
        self.sampleFOV_list.setFrameShape(QFrame.Box)
        self.sampleFOV_list.setFrameShadow(QFrame.Sunken)
        self.sample_FOVlist_layout = QVBoxLayout(self.sampleFOV_list)
        self.sample_FOVlist_layout.setObjectName(u"sample_FOVlist_layout")
        self.sampleFOV1 = QLabel(self.sampleFOV_list)
        self.sampleFOV1.setObjectName(u"sampleFOV1")
        self.sampleFOV1.setText(u"")

        self.sample_FOVlist_layout.addWidget(self.sampleFOV1)

        self.sampleFOV2 = QLabel(self.sampleFOV_list)
        self.sampleFOV2.setObjectName(u"sampleFOV2")

        self.sample_FOVlist_layout.addWidget(self.sampleFOV2)

        self.sampleFOV3 = QLabel(self.sampleFOV_list)
        self.sampleFOV3.setObjectName(u"sampleFOV3")
        self.sampleFOV3.setText(u"")

        self.sample_FOVlist_layout.addWidget(self.sampleFOV3)

        self.sampleFOV4 = QLabel(self.sampleFOV_list)
        self.sampleFOV4.setObjectName(u"sampleFOV4")
        self.sampleFOV4.setText(u"")

        self.sample_FOVlist_layout.addWidget(self.sampleFOV4)

        self.sampleFOV5 = QLabel(self.sampleFOV_list)
        self.sampleFOV5.setObjectName(u"sampleFOV5")
        self.sampleFOV5.setText(u"")

        self.sample_FOVlist_layout.addWidget(self.sampleFOV5)


        self.sampleFOV_layout.addWidget(self.sampleFOV_list)


        self.samples_layout_2.addWidget(self.samplesFOV_widget)


        self.container_layout.addWidget(self.samples_widget)

        self.buttonBox = QDialogButtonBox(self.container)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Close|QDialogButtonBox.Ok|QDialogButtonBox.Reset)

        self.container_layout.addWidget(self.buttonBox)


        self.horizontalLayout.addWidget(self.container)


        self.retranslateUi(FOV2ROIlinks)
        self.buttonBox.clicked.connect(FOV2ROIlinks.button_clicked)
        self.pos_right.currentTextChanged.connect(FOV2ROIlinks.change_sample_style)
        self.pos_left.currentTextChanged.connect(FOV2ROIlinks.change_sample_style)
        self.pos_pattern.currentTextChanged.connect(FOV2ROIlinks.change_sample_style)
        self.pos_colour.clicked.connect(FOV2ROIlinks.choose_colour)

        QMetaObject.connectSlotsByName(FOV2ROIlinks)
    # setupUi

    def retranslateUi(self, FOV2ROIlinks):
        FOV2ROIlinks.setWindowTitle(QCoreApplication.translate("FOV2ROIlinks", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("FOV2ROIlinks", u"FOV name pattern", None))
        self.pos_colour.setText("")
        self.sampleFOV2.setText("")
    # retranslateUi

