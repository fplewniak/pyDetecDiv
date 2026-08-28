"""
Generic widgets providing basic functionalities to build tool GUIs. These widgets are expected to be extended and implemented
to meet the specific needs of new_tools
"""
from typing import Any, Callable, cast

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QMenu

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.core.widgets.threading import StdoutWaitDialog, WaitDialog, PyDetecDivThread
from pydetecdiv.app.gui.core.widgets import Dialog
from pydetecdiv.app.tools import Tool


class ToolDialog(Dialog):
    """
    Generic tool dialog window
    """
    progress = Signal(int)
    finished = Signal(object)

    def __init__(self, tool: Tool, title: str | None = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(title, **kwargs)
        self.tool = cast(type[tool], tool)
        self.post_command_func = []
        self.finished.connect(self.on_finished)

    def wait_for_command(self, msg: str | None = None, cancel_msg: str | None = None) -> None:
        """
        Launch the conversion and wait for completion
        """
        wait_dialog = WaitDialog(msg, self, title=self.tool.title, cancel_msg=cancel_msg, progress_bar=True, )
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            project.back_up()
        wait_dialog.wait_for(self.run_command_with_progress)

    def run_command_with_progress(self, thread: PyDetecDivThread | None = None):
        """
        Run a command sending progress and finished signals
        """
        try:
            for i in self.tool.callback():
                if thread and thread.isInterruptionRequested():
                    self.finished.emit(False)  # False = cancelled
                    return
                self.progress.emit(i)
            self.finished.emit(True)  # True = success
        except Exception as e:
            self.finished.emit(e)  # Exception = error

    def run_command_with_stdout(self, func: Callable, title: str, close_when_finished: bool = True, **kwargs) -> None:
        """
        Open a waiting dialog window to wait for completion of job
        """
        wait_dialog = StdoutWaitDialog(title, self, close_when_finished=close_when_finished)
        wait_dialog.resize(500, 300)
        self.finished.connect(wait_dialog.stop_redirection)
        wait_dialog.wait_for(lambda: self.run_process(func), **kwargs)
        self.close()

    def run_after_process(self, list_func: list[Callable]) -> None:
        """
        Declares the list of functions that should be run when the job is finished
        :param list_func: the list of functions
        """
        self.post_command_func.extend(list_func)

    def run_process(self, func: Callable, **kwargs) -> None:
        """
        Run a job
        """
        self.finished.emit(func(**kwargs))

    def on_finished(self, success):
        if self.tool.run:
            print(f"Run {self.tool.run.id_}")
        if success is False:
            print("Job was cancelled")
            self.tool.cancel_run()
            if self.tool.rollback is not None:
                self.tool.rollback()
        elif isinstance(success, Exception):
            print(f"Job failed: {success}")
            self.tool.cancel_run()
            if self.tool.rollback is not None:
                self.tool.rollback()
        else:
            with pydetecdiv_project(PyDetecDiv.project_name) as project:
                project.delete_backup()
            for func in self.post_command_func:
                func()
            print("Job completed successfully")

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        When the Dialog is closed, undeclare parameters to save in the run record (they should have been saved already anyway)
        :param event: the close event
        """
        for parameter in self.tool.parameters.parameter_list:
            parameter.should_be_saved = False


class ToolMenu(QMenu):
    """
    Abstract Tool submenu class
    """

    def __init__(self, tool_name: str, enable: Callable | None = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.tool = PyDetecDiv.tools[tool_name]
        self.setTitle(self.tool.name)
        self.enabling_function = enable
        self._parent = None

    def add_to_menu(self, menu: QMenu) -> None:
        """
        Add the menu to another one as a submenu

        :param menu: the menu to add the submenu to
        """
        menu.addMenu(self)
        self._parent = menu
        menu.aboutToShow.connect(self.determine_enabled_status)

    def determine_enabled_status(self):
        """
        Enable or disable the menu depending on the enabling function output

        :param kwargs: extra keyword arguments
        """
        try:
            self.setEnabled(self.enabling_function())
        except TypeError as e:
            self.setEnabled(True)


class ToolAction(QAction):
    """
    Generic action to trigger a tool process
    """

    def __init__(self, tool_name: str, command: str, launch: Callable,  parent: QMenu | None = None,
                 enable: Callable | None = None, checkable: bool = False):
        self.tool = PyDetecDiv.tools[tool_name]
        self.command = command
        super().__init__(self.tool.commands[command].title, parent)

        self.enabling_function = enable
        self.launch_callable = launch
        self.triggered.connect(self.launch)
        self._parent = None
        if parent is not None:
            self.add_to_menu(parent)
        self.setCheckable(checkable)

    def add_to_menu(self, menu: QMenu) -> None:
        """
        Add the action to a menu

        :param menu: the menu to add the action to
        """
        menu.addAction(self)
        self._parent = menu
        menu.aboutToShow.connect(self.determine_enabled_status)

    def parent(self) -> QMenu:
        """
        Returns the parent menu
        :return: the parent ToolMenu
        """
        return self._parent

    def determine_enabled_status(self):
        """
        Enable or disable the action depending on the enabling function output

        :param kwargs: extra keyword arguments
        """
        try:
            self.setEnabled(self.enabling_function())
        except TypeError as e:
            self.setEnabled(True)

    def launch(self):
        """
        Launch the action
        """
        self.tool.command = self.command
        self.launch_callable(self.tool)
