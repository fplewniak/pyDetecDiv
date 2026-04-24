"""
Generic widgets providing basic functionalities to build tool GUIs. These widgets are expected to be extended and implemented
to meet the specific needs of new_tools
"""
from abc import abstractmethod
from typing import Any, Callable

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QMenu

from pydetecdiv.app import StdoutWaitDialog
from pydetecdiv.app.gui.core.widgets import Dialog
from pydetecdiv.app.tools import Tool


class ToolDialog(Dialog):
    """
    Generic tool dialog window
    """
    job_finished: Signal = Signal(object)

    def __init__(self, tool: Tool, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(title, **kwargs)
        self.tool = tool

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        When the Dialog is closed, undeclare parameters to save in the run record (they should have been saved already anyway)
        :param event: the close event
        """
        for parameter in self.tool.parameters.parameter_list:
            parameter.should_be_saved = False

    def wait_for_process(self, func: Callable) -> None:
        """
        Open a waiting dialog window to wait for completion of job
        """
        wait_dialog = StdoutWaitDialog('**Training model**', self)
        wait_dialog.resize(500, 300)
        self.job_finished.connect(wait_dialog.stop_redirection)
        wait_dialog.wait_for(lambda: self.run_process(func))
        self.close()

    def run_after_process(self, list_func: list[Callable]) -> None:
        """
        Declares the list of functions that should be run when the job is finished
        :param list_func: the list of functions
        """
        for func in list_func:
            self.job_finished.connect(func)

    def run_process(self, func) -> None:
        """
        Run a job
        """
        self.job_finished.emit(func())


class ToolMenu(QMenu):
    """
    Abstract Tool submenu class
    """
    def __init__(self, tool: Tool, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.tool = tool

class ToolAction(QAction):
    """
    Generic action to trigger a tool process
    """
    def __init__(self, title: str, parent: ToolMenu, **kwargs: dict[str, Any]) -> None:
        super().__init__(title, parent)
        self.triggered.connect(self.launch)
        self.setEnabled(False)
        parent.addAction(self)
        self._parent = parent

    def parent(self) -> ToolMenu:
        """
        Returns the parent menu
        :return: the parent ToolMenu
        """
        return self._parent

    @abstractmethod
    def launch(self):
        """
        Launch the action
        """
