"""
GUI for an example plugin showing how to interact with the tabbed viewer
"""
from PySide6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox


class AddPlotDialog(QDialog):
    """
    Action dialog window: OK button launches the action
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.layout = QVBoxLayout(self)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Close | QDialogButtonBox.Ok, self)
        self.button_box.setCenterButtons(True)
        self.button_box.rejected.connect(self.close)
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)
