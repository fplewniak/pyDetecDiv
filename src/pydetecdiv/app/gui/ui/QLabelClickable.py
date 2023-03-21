"""
Custom clickable QLabel
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel


class QLabelClickable(QLabel):
    """
    A class extending the QLabel class to make label clickable
    """
    clicked = Signal(str)

    def mousePressEvent(self, ev):
        """
        Reaction to mouse click.
        :param ev: the click event
        """
        self.clicked.emit(self.objectName())
