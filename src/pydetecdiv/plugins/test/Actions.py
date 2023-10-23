from PySide6.QtGui import QAction
from pydetecdiv.plugins.example.ActionDialog import ActionDialog


class Action1(QAction):
    def __init__(self, parent):
        super().__init__("Test action1", parent)
        self.triggered.connect(self.action)
        self.triggered.connect(ActionDialog)
        parent.addAction(self)

    def action(self):
        print('run test action1')


class Action2(QAction):
    def __init__(self, parent):
        super().__init__("Test action2", parent)
        self.triggered.connect(self.action)
        parent.addAction(self)

    def action(self):
        print('run test action2')
