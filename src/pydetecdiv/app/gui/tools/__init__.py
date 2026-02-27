"""
Generic widgets providing basic functionalities to build tool GUIs. These widgets are expected to be extended and implemented
to meet the specific needs of tools
"""
from abc import abstractmethod
from typing import Any

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from pydetecdiv.app.gui.core.widgets import Dialog
from pydetecdiv.app.tools import Tool


class ToolDialog(Dialog):
    """
    Generic tool dialog window
    """
    def __init__(self, tool: Tool, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(title, **kwargs)
        self.tool = tool


class ToolMenu(QMenu):
    """
    Abstract Tool submenu class
    """
    def __init__(self, tool: Tool, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.tool = tool

class ToolAction(QAction):
    def __init__(self, title: str, parent: ToolMenu, **kwargs: dict[str, Any]) -> None:
        super().__init__(title, parent)
        self.triggered.connect(self.launch)
        self.setEnabled(False)
        parent.addAction(self)
        self._parent = parent

    def parent(self) -> ToolMenu:
        return self._parent

    @abstractmethod
    def launch(self):
        """
        Launch the action
        """
