from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel


class QLabelClickable(QLabel):
    clicked=Signal(str)

    def mousePressEvent(self, ev):
        self.clicked.emit(self.objectName())
