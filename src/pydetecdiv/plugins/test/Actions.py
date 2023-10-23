from PySide6.QtGui import QAction

from pydetecdiv.plugins.test.ActionDialog import ActionDialog


class Action1(QAction):
    def __init__(self, parent, plugin):
        super().__init__("Test action1", parent)
        self.triggered.connect(self.action)
        self.triggered.connect(ActionDialog)
        self.plugin = plugin
        parent.addAction(self)

    def action(self):
        print('run test action1')


class Action2(QAction):
    def __init__(self, parent, plugin):
        super().__init__("Change pen", parent)
        self.plugin = plugin
        self.triggered.connect(self.plugin.change_pen)
        parent.addAction(self)