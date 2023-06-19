# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'RawData2FOV.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QMetaObject, QSize, Qt)
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialogButtonBox, QFrame, QHBoxLayout,
                               QLabel, QSizePolicy, QVBoxLayout, QWidget)

from pydetecdiv.app.gui.ui.QLabelClickable import QLabelClickable

class Ui_RawData2FOV(object):
    def setupUi(self, RawData2FOV):
        if not RawData2FOV.objectName():
            RawData2FOV.setObjectName(u"RawData2FOV")
        RawData2FOV.resize(599, 361)
        self.horizontalLayout = QHBoxLayout(RawData2FOV)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.container = QWidget(RawData2FOV)
        self.container.setObjectName(u"container")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pos_widget.sizePolicy().hasHeightForWidth())
        self.pos_widget.setSizePolicy(sizePolicy1)
        self.pos_layout = QHBoxLayout(self.pos_widget)
        self.pos_layout.setObjectName(u"pos_layout")
        self.pos_layout.setContentsMargins(0, 0, 0, 0)
        self.pos_check = QCheckBox(self.pos_widget)
        self.pos_check.setObjectName(u"pos_check")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.pos_check.sizePolicy().hasHeightForWidth())
        self.pos_check.setSizePolicy(sizePolicy2)

        self.pos_layout.addWidget(self.pos_check)

        self.pos_left = QComboBox(self.pos_widget)
        self.pos_left.setObjectName(u"pos_left")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        sizePolicy4 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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

        self.c_widget = QWidget(self.patterns_widget)
        self.c_widget.setObjectName(u"c_widget")
        sizePolicy1.setHeightForWidth(self.c_widget.sizePolicy().hasHeightForWidth())
        self.c_widget.setSizePolicy(sizePolicy1)
        self.c_layout = QHBoxLayout(self.c_widget)
        self.c_layout.setSpacing(6)
        self.c_layout.setObjectName(u"c_layout")
        self.c_layout.setContentsMargins(0, 0, 0, 0)
        self.c_check = QCheckBox(self.c_widget)
        self.c_check.setObjectName(u"c_check")
        sizePolicy2.setHeightForWidth(self.c_check.sizePolicy().hasHeightForWidth())
        self.c_check.setSizePolicy(sizePolicy2)

        self.c_layout.addWidget(self.c_check)

        self.c_left = QComboBox(self.c_widget)
        self.c_left.setObjectName(u"c_left")
        sizePolicy3.setHeightForWidth(self.c_left.sizePolicy().hasHeightForWidth())
        self.c_left.setSizePolicy(sizePolicy3)
        self.c_left.setEditable(True)
        self.c_left.setCurrentText(u"")

        self.c_layout.addWidget(self.c_left)

        self.c_pattern = QComboBox(self.c_widget)
        self.c_pattern.setObjectName(u"c_pattern")
        sizePolicy3.setHeightForWidth(self.c_pattern.sizePolicy().hasHeightForWidth())
        self.c_pattern.setSizePolicy(sizePolicy3)
        self.c_pattern.setStyleSheet(u"border: 2px solid rgb(0, 255, 0);")
        self.c_pattern.setEditable(True)
        self.c_pattern.setCurrentText(u"")

        self.c_layout.addWidget(self.c_pattern)

        self.c_right = QComboBox(self.c_widget)
        self.c_right.setObjectName(u"c_right")
        sizePolicy3.setHeightForWidth(self.c_right.sizePolicy().hasHeightForWidth())
        self.c_right.setSizePolicy(sizePolicy3)
        self.c_right.setEditable(True)

        self.c_layout.addWidget(self.c_right)

        self.c_colour = QLabelClickable(self.c_widget)
        self.c_colour.setObjectName(u"c_colour")
        sizePolicy4.setHeightForWidth(self.c_colour.sizePolicy().hasHeightForWidth())
        self.c_colour.setSizePolicy(sizePolicy4)
        self.c_colour.setMinimumSize(QSize(20, 20))
        self.c_colour.setStyleSheet(u"background-color: rgb(0, 255, 0);")
        self.c_colour.setFrameShape(QFrame.Panel)
        self.c_colour.setFrameShadow(QFrame.Sunken)
        self.c_colour.setLineWidth(3)

        self.c_layout.addWidget(self.c_colour)


        self.patterns_layout.addWidget(self.c_widget)

        self.t_widget = QWidget(self.patterns_widget)
        self.t_widget.setObjectName(u"t_widget")
        sizePolicy1.setHeightForWidth(self.t_widget.sizePolicy().hasHeightForWidth())
        self.t_widget.setSizePolicy(sizePolicy1)
        self.t_layout = QHBoxLayout(self.t_widget)
        self.t_layout.setObjectName(u"t_layout")
        self.t_layout.setContentsMargins(0, 0, 0, 0)
        self.t_check = QCheckBox(self.t_widget)
        self.t_check.setObjectName(u"t_check")
        sizePolicy2.setHeightForWidth(self.t_check.sizePolicy().hasHeightForWidth())
        self.t_check.setSizePolicy(sizePolicy2)
        self.t_check.setMinimumSize(QSize(70, 20))

        self.t_layout.addWidget(self.t_check)

        self.t_left = QComboBox(self.t_widget)
        self.t_left.setObjectName(u"t_left")
        sizePolicy3.setHeightForWidth(self.t_left.sizePolicy().hasHeightForWidth())
        self.t_left.setSizePolicy(sizePolicy3)
        self.t_left.setEditable(True)
        self.t_left.setCurrentText(u"")

        self.t_layout.addWidget(self.t_left)

        self.t_pattern = QComboBox(self.t_widget)
        self.t_pattern.setObjectName(u"t_pattern")
        sizePolicy3.setHeightForWidth(self.t_pattern.sizePolicy().hasHeightForWidth())
        self.t_pattern.setSizePolicy(sizePolicy3)
        self.t_pattern.setStyleSheet(u"border: 2px solid rgb(0, 255, 255);")
        self.t_pattern.setEditable(True)
        self.t_pattern.setCurrentText(u"")

        self.t_layout.addWidget(self.t_pattern)

        self.t_right = QComboBox(self.t_widget)
        self.t_right.setObjectName(u"t_right")
        sizePolicy3.setHeightForWidth(self.t_right.sizePolicy().hasHeightForWidth())
        self.t_right.setSizePolicy(sizePolicy3)
        self.t_right.setEditable(True)

        self.t_layout.addWidget(self.t_right)

        self.t_colour = QLabelClickable(self.t_widget)
        self.t_colour.setObjectName(u"t_colour")
        sizePolicy4.setHeightForWidth(self.t_colour.sizePolicy().hasHeightForWidth())
        self.t_colour.setSizePolicy(sizePolicy4)
        self.t_colour.setMinimumSize(QSize(20, 20))
        self.t_colour.setStyleSheet(u"background-color: rgb(0, 255, 255);")
        self.t_colour.setFrameShape(QFrame.Panel)
        self.t_colour.setFrameShadow(QFrame.Sunken)
        self.t_colour.setLineWidth(3)

        self.t_layout.addWidget(self.t_colour)


        self.patterns_layout.addWidget(self.t_widget)

        self.z_widget = QWidget(self.patterns_widget)
        self.z_widget.setObjectName(u"z_widget")
        sizePolicy1.setHeightForWidth(self.z_widget.sizePolicy().hasHeightForWidth())
        self.z_widget.setSizePolicy(sizePolicy1)
        self.z_layout = QHBoxLayout(self.z_widget)
        self.z_layout.setObjectName(u"z_layout")
        self.z_layout.setContentsMargins(0, 0, 0, 0)
        self.z_check = QCheckBox(self.z_widget)
        self.z_check.setObjectName(u"z_check")
        sizePolicy2.setHeightForWidth(self.z_check.sizePolicy().hasHeightForWidth())
        self.z_check.setSizePolicy(sizePolicy2)
        self.z_check.setMinimumSize(QSize(70, 20))

        self.z_layout.addWidget(self.z_check)

        self.z_left = QComboBox(self.z_widget)
        self.z_left.setObjectName(u"z_left")
        sizePolicy3.setHeightForWidth(self.z_left.sizePolicy().hasHeightForWidth())
        self.z_left.setSizePolicy(sizePolicy3)
        self.z_left.setEditable(True)
        self.z_left.setCurrentText(u"")

        self.z_layout.addWidget(self.z_left)

        self.z_pattern = QComboBox(self.z_widget)
        self.z_pattern.setObjectName(u"z_pattern")
        sizePolicy3.setHeightForWidth(self.z_pattern.sizePolicy().hasHeightForWidth())
        self.z_pattern.setSizePolicy(sizePolicy3)
        self.z_pattern.setStyleSheet(u"border: 2px solid rgb(255, 255, 0);")
        self.z_pattern.setEditable(True)
        self.z_pattern.setCurrentText(u"")

        self.z_layout.addWidget(self.z_pattern)

        self.z_right = QComboBox(self.z_widget)
        self.z_right.setObjectName(u"z_right")
        sizePolicy3.setHeightForWidth(self.z_right.sizePolicy().hasHeightForWidth())
        self.z_right.setSizePolicy(sizePolicy3)
        self.z_right.setEditable(True)

        self.z_layout.addWidget(self.z_right)

        self.z_colour = QLabelClickable(self.z_widget)
        self.z_colour.setObjectName(u"z_colour")
        sizePolicy4.setHeightForWidth(self.z_colour.sizePolicy().hasHeightForWidth())
        self.z_colour.setSizePolicy(sizePolicy4)
        self.z_colour.setMinimumSize(QSize(20, 20))
        self.z_colour.setStyleSheet(u"background-color: rgb(255, 255, 0);")
        self.z_colour.setFrameShape(QFrame.Panel)
        self.z_colour.setFrameShadow(QFrame.Sunken)
        self.z_colour.setLineWidth(3)

        self.z_layout.addWidget(self.z_colour)


        self.patterns_layout.addWidget(self.z_widget)


        self.verticalLayout.addWidget(self.patterns_widget)


        self.container_layout.addWidget(self.control_widgets)

        self.samples_widget = QWidget(self.container)
        self.samples_widget.setObjectName(u"samples_widget")
        sizePolicy5 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.samples_widget.sizePolicy().hasHeightForWidth())
        self.samples_widget.setSizePolicy(sizePolicy5)
        self.samples_layout = QVBoxLayout(self.samples_widget)
        self.samples_layout.setObjectName(u"samples_layout")
        self.samples_layout.setContentsMargins(-1, 0, -1, -1)
        self.samples_box_title = QLabel(self.samples_widget)
        self.samples_box_title.setObjectName(u"samples_box_title")
        sizePolicy6 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.samples_box_title.sizePolicy().hasHeightForWidth())
        self.samples_box_title.setSizePolicy(sizePolicy6)
        self.samples_box_title.setStyleSheet(u"font: bold;")
        self.samples_box_title.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.samples_layout.addWidget(self.samples_box_title)

        self.sample_list = QFrame(self.samples_widget)
        self.sample_list.setObjectName(u"sample_list")
        sizePolicy7 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.sample_list.sizePolicy().hasHeightForWidth())
        self.sample_list.setSizePolicy(sizePolicy7)
        self.sample_list.setFrameShape(QFrame.Box)
        self.sample_list.setFrameShadow(QFrame.Sunken)
        self.sample_list_layout = QVBoxLayout(self.sample_list)
        self.sample_list_layout.setObjectName(u"sample_list_layout")
        self.sample1 = QLabel(self.sample_list)
        self.sample1.setObjectName(u"sample1")
        sizePolicy8 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.sample1.sizePolicy().hasHeightForWidth())
        self.sample1.setSizePolicy(sizePolicy8)

        self.sample_list_layout.addWidget(self.sample1)

        self.sample2 = QLabel(self.sample_list)
        self.sample2.setObjectName(u"sample2")
        sizePolicy8.setHeightForWidth(self.sample2.sizePolicy().hasHeightForWidth())
        self.sample2.setSizePolicy(sizePolicy8)

        self.sample_list_layout.addWidget(self.sample2)

        self.sample3 = QLabel(self.sample_list)
        self.sample3.setObjectName(u"sample3")
        sizePolicy8.setHeightForWidth(self.sample3.sizePolicy().hasHeightForWidth())
        self.sample3.setSizePolicy(sizePolicy8)

        self.sample_list_layout.addWidget(self.sample3)

        self.sample4 = QLabel(self.sample_list)
        self.sample4.setObjectName(u"sample4")
        sizePolicy8.setHeightForWidth(self.sample4.sizePolicy().hasHeightForWidth())
        self.sample4.setSizePolicy(sizePolicy8)

        self.sample_list_layout.addWidget(self.sample4)

        self.sample5 = QLabel(self.sample_list)
        self.sample5.setObjectName(u"sample5")
        sizePolicy8.setHeightForWidth(self.sample5.sizePolicy().hasHeightForWidth())
        self.sample5.setSizePolicy(sizePolicy8)

        self.sample_list_layout.addWidget(self.sample5)


        self.samples_layout.addWidget(self.sample_list)


        self.container_layout.addWidget(self.samples_widget)

        self.buttonBox = QDialogButtonBox(self.container)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok|QDialogButtonBox.Reset)

        self.container_layout.addWidget(self.buttonBox)


        self.horizontalLayout.addWidget(self.container)


        self.retranslateUi(RawData2FOV)
        self.pos_check.stateChanged.connect(RawData2FOV.change_sample_style)
        self.c_check.stateChanged.connect(RawData2FOV.change_sample_style)
        self.t_check.stateChanged.connect(RawData2FOV.change_sample_style)
        self.z_check.stateChanged.connect(RawData2FOV.change_sample_style)
        self.pos_left.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.c_left.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.t_left.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.z_left.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.pos_pattern.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.c_pattern.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.t_pattern.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.z_pattern.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.pos_right.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.c_right.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.t_right.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.z_right.currentTextChanged.connect(RawData2FOV.change_sample_style)
        self.pos_colour.clicked.connect(RawData2FOV.choose_colour)
        self.c_colour.clicked.connect(RawData2FOV.choose_colour)
        self.t_colour.clicked.connect(RawData2FOV.choose_colour)
        self.z_colour.clicked.connect(RawData2FOV.choose_colour)
        self.buttonBox.clicked.connect(RawData2FOV.button_clicked)

        QMetaObject.connectSlotsByName(RawData2FOV)
    # setupUi

    def retranslateUi(self, RawData2FOV):
        RawData2FOV.setWindowTitle(QCoreApplication.translate("RawData2FOV", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("RawData2FOV", u"File name patterns", None))
        self.pos_check.setText(QCoreApplication.translate("RawData2FOV", u"Position", None))
        self.pos_colour.setText("")
        self.c_check.setText(QCoreApplication.translate("RawData2FOV", u"Channel", None))
        self.c_colour.setText("")
        self.t_check.setText(QCoreApplication.translate("RawData2FOV", u"Frame", None))
        self.t_colour.setText("")
        self.z_check.setText(QCoreApplication.translate("RawData2FOV", u"Layer", None))
        self.z_colour.setText("")
        self.samples_box_title.setText(QCoreApplication.translate("RawData2FOV", u"Sample file names", None))
    # retranslateUi

