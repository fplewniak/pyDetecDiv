from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

from pydetecdiv.app import PyDetecDiv

class ActionDialog(QDialog):
    def __init__(self):
        super().__init__(PyDetecDiv().main_window)
        self.setMinimumWidth(200)
        layout = QVBoxLayout(self)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Close | QDialogButtonBox.Ok, self)
        self.button_box.setCenterButtons(True)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
        self.exec()

    def accept(self):
        print('OK')
