from PySide6.QtGui import QAction

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.plugins.example.ActionDockWindow import ActionDockWindow


class Action1(QAction):
    def __init__(self, parent):
        super().__init__("Example action1", parent)
        self.triggered.connect(self.action)
        self.gui = None
        parent.addAction(self)

    def action(self):
        print('run example action1')
        self.gui = ActionDockWindow()
        self.gui.setVisible(True)


class Action2(QAction):
    def __init__(self, parent):
        super().__init__("Example action2", parent)
        self.triggered.connect(self.action)
        parent.addAction(self)

    def action(self):
        print(f'run example action2 {PyDetecDiv().project_name}')
