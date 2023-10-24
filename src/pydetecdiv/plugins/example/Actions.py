"""
Actions to launch functions added by the Example plugin
"""
from PySide6.QtGui import QAction

from pydetecdiv.app import PyDetecDiv


class Action1(QAction):
    """
    Class defining a QAction to insert into the Example menu. This action will launch a method in the Plugin object
    creating a new table and then saving results in the database.
    """
    def __init__(self, parent):
        super().__init__("Create and save results", parent)
        parent.addAction(self)


class Action2(QAction):
    """
    Class defining a QAction to insert into the Example menu
    """
    def __init__(self, parent):
        super().__init__("Example action2", parent)
        self.triggered.connect(self.action)
        parent.addAction(self)

    def action(self):
        """
        Method running code when Action2 has been triggered
        """
        print(f'run example action2 {PyDetecDiv().project_name}')
