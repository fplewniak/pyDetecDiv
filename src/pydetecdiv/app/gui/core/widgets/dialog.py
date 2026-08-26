from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QDialogButtonBox


class MessageDialog(QDialog):
    """
    Generic dialog to communicate a message to the user (error, warning or any other information)
    """

    def __init__(self, msg: str, html: bool = True):
        super().__init__()
        # self.setWindowModality(Qt.WindowModal)
        label = QLabel()
        label.setText(msg)
        if html:
            label.setTextFormat(Qt.TextFormat.RichText)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)
        self.setLayout(layout)
        self.exec()


class ConfirmDialog(QDialog):
    """
    Generic dialog asking for confirmation from the user to launch an action
    """

    def __init__(self, msg: str, action: Callable):
        super().__init__()
        # self.setWindowModality(Qt.WindowModal)
        self.action = action
        label = QLabel()
        # label.setStyleSheet("""
        # font-weight: bold;
        # """)
        label.setText(msg)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)
        self.setLayout(layout)
        self.exec()

    def accept(self, /):
        """
        Close the window and launch action
        """
        self.close()
        self.action()
