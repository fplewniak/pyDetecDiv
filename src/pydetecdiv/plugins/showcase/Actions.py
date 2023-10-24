"""
Actions for the showcase plugin
"""
from PySide6.QtGui import QAction

from pydetecdiv.plugins.showcase.ActionDialog import ActionDialog


class Action1(QAction):
    """
    Action to open a dialog window that will add a plot to the currently active subwindow when the OK button is clicked
    """
    def __init__(self, parent, plugin):
        super().__init__("Plot dialog window", parent)
        self.triggered.connect(lambda _: ActionDialog(plugin))
        self.plugin = plugin
        parent.addAction(self)


class Action2(QAction):
    """
    Action to change the style of the pen
    """
    def __init__(self, parent, plugin):
        super().__init__("Change pen", parent)
        self.plugin = plugin
        self.triggered.connect(self.plugin.change_pen)
        parent.addAction(self)

class Action3(QAction):
    """
    Action to add a plot to the currently active subwindow
    """
    def __init__(self, parent, plugin):
        super().__init__("Add plot", parent)
        self.triggered.connect(plugin.add_plot)
        self.plugin = plugin
        parent.addAction(self)
