import sys
from typing import Callable

import markdown
from PySide6.QtCore import QThread, Slot, Qt, Signal, QObject
from PySide6.QtGui import QCursor, QCloseEvent, QTextCursor
from PySide6.QtWidgets import QDialog, QWidget, QLabel, QVBoxLayout, QProgressBar, QDialogButtonBox, QTextEdit

from pydetecdiv.app import PyDetecDiv


class PyDetecDivThread(QThread):
    """
    Thread used to run a process defined by a function and its arguments
    """

    def __init__(self):
        super().__init__()
        self.func: Callable | None = None
        self.args: list = []
        self.kwargs: dict = {}

    def set_function(self, func: Callable, *args: list, **kwargs: dict) -> None:
        """
        Define the function to run in the thread

        :param func: the function to run
        :param args: arguments passed to the function
        :param kwargs: keyword arguments passed to the function
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self) -> None:
        """
        Run the function
        """
        if self.func is not None:
            self.func(*self.args, thread=self, **self.kwargs)


class AbstractWaitDialog(QDialog):
    """
    Generic dialog box asking the user to wait for a thread to be finished. This box closes automatically when the
    thread is complete and the parent window is hidden as well if it is specified. This should be used for processes
    that do not last too long and that might generate inconsistency if cancelled as there is no possibility to interrupt
    it
    """

    def __init__(self, parent: QWidget, title: str | None = None, cancel_msg: str | None = None,
                 ignore_close_event: bool= True, close_when_finished: bool = True) -> None:
        super().__init__(parent)
        if title is not None:
            self.setWindowTitle(title)
        else:
            self.setWindowTitle(PyDetecDiv.project_name)
        self.cancel_msg = cancel_msg
        self._ignore_close_event = ignore_close_event
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.pdd_thread = PyDetecDivThread()
        self.parent = parent
        if hasattr(self.parent, 'finished') and close_when_finished:
            self.parent.finished.connect(self.close_window)

    def wait_for(self, func: Callable, *args: list, **kwargs: dict) -> None:
        """
        Run function in separate thread and launch local event loop to handle progress bar and cancellation

        :param func: the function to run
        :param args: positional arguments for the function
        :param kwargs: keyword arguments for the function
        """
        PyDetecDiv.app.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        self.pdd_thread.set_function(func, *args, **kwargs)
        self.pdd_thread.start()
        self.exec()

    def close_window(self) -> None:
        """
        Hide and destroy the Wait dialog window. The cursor is also set back to its normal aspect.
        """
        self.hide()
        PyDetecDiv.app.restoreOverrideCursor()
        if self.pdd_thread.isRunning():
            self.pdd_thread.terminate()
        self.destroy()

    def cancel(self) -> None:
        """
        Set cancelling message and request for interruption of thread so that the running job can cleanly close
        processes and roll back any modification if needed.
        """
        if self.pdd_thread.isRunning():
            self.pdd_thread.requestInterruption()

    def set_ignore_close_event(self, ignore_close_event: bool = True) -> None:
        """
        Set the _ignore_close_event flag to prevent or allow closing the window

        :param ignore_close_event: value to set the flag to
        """
        self._ignore_close_event = ignore_close_event

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Cancel the job if the window is closed unless close event is ignored by request.

        :param event: close event
        """
        if self._ignore_close_event:
            event.ignore()
        else:
            self.cancel()


class WaitDialog(AbstractWaitDialog):
    """
    Generic dialog box asking the user to wait for a thread to be finished. This box closes automatically when the
    thread is complete and the parent window is hidden as well if it is specified. This should be used for processes
    that do not last too long and that might generate inconsistency if cancelled as there is no possibility to interrupt
    it
    """

    def __init__(self, msg, parent: QWidget, title: str | None = None, progress_bar: bool = False, cancel_msg: str | None= None,
                 ignore_close_event: bool = True, close_when_finished: bool = True):
        super().__init__(parent, title=title, cancel_msg=cancel_msg, ignore_close_event=ignore_close_event,
                         close_when_finished=close_when_finished)
        if hasattr(self.parent, 'progress'):
            self.parent.progress.connect(self.show_progress)

        self.label = QLabel()
        # self.label.setStyleSheet("""
        # font-weight: bold;
        # """)
        self.label.setText(msg)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        if progress_bar:
            self.progress_bar_widget = QProgressBar()
            layout.addWidget(self.progress_bar_widget)
        if cancel_msg:
            self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
            self.button_box.rejected.connect(self.cancel)
            # self.button_box.rejected.connect(self.button_box.hide)
            # self.button_box.rejected.connect(self.set_ignore_close_event)
            layout.addWidget(self.button_box)
        self.setLayout(layout)

    def show_progress(self, i: int) -> None:
        """
        Convenience method to send the progress value to the progress bar widget

        :param i: the value to pass to the progress bar
        """
        self.progress_bar_widget.setValue(i)

    def cancel(self) -> None:
        """
        Set cancelling message and request for interruption of thread so that the running job can cleanly close
        processes and roll back any modification if needed.
        """
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setEnabled(False)
        self.set_ignore_close_event(True)
        if self.cancel_msg:
            self.label.setText(self.cancel_msg)
        super().cancel()
        # self.close_window()


class StdoutWaitDialog(AbstractWaitDialog):
    """
    A Wait dialog that also captures and displays stdout output on the fly.
    """

    def __init__(self, msg: str, parent: QWidget, cancel_msg: str | None = None, ignore_close_event: bool = True,
                 close_when_finished: bool = True):
        super().__init__(parent, cancel_msg=cancel_msg, ignore_close_event=ignore_close_event,
                         close_when_finished=close_when_finished)
        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setHtml(markdown.markdown(msg))
        layout = QVBoxLayout(self)
        layout.addWidget(self.log)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.button_box.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close_window)
        self.button_box.button(QDialogButtonBox.StandardButton.Close).setEnabled(False)
        if self.cancel_msg:
            self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
            self.button_box.button(QDialogButtonBox.StandardButton.Cancel).clicked.connect(self.cancel)
            self.button_box.button(QDialogButtonBox.StandardButton.Cancel).clicked.connect(self.set_ignore_close_event)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
        self.redirector = StreamRedirector()
        self.redirector.new_text.connect(self.addHtmlText)
        sys.stdout = self.redirector

    def addText(self, text: str) -> None:
        """
        Add text to the log window

        :param text: the text to add
        """
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        if hasattr(self.parent, 'tool'):
            self.parent.tool.log_text(text)

    def addHtmlText(self, text: str) -> None:
        """
        Add text to the log window

        :param text: the text to add
        """
        html = markdown.markdown(text, extensions=['tables'])
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertHtml(html)
        self.log.insertHtml('<br>')
        if hasattr(self.parent, 'tool'):
            self.parent.tool.log_text(text)

    def cancel(self) -> None:
        """
        Set cancelling message and request for interruption of thread so that the running job can cleanly close
        processes and roll back any modification if needed.
        """
        if self.cancel_msg:
            self.log.append(self.cancel_msg)
        super().cancel()

    def close_window(self) -> None:
        """
        Closes the window and stops stdout capture
        """
        sys.stdout = sys.__stdout__
        super().close_window()

    def stop_redirection(self, signal: Signal) -> None:
        """
        Stops capturing the stdout output, which is therefore printed to the terminal again

        :param signal: the signal triggered by the event requesting to stop redirection
        """
        if self.cancel_msg:
            self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setEnabled(False)
        self.button_box.button(QDialogButtonBox.StandardButton.Close).setEnabled(True)
        PyDetecDiv.app.restoreOverrideCursor()
        sys.stdout = sys.__stdout__


class StreamRedirector(QObject):
    """Custom stream redirector to emit stdout/stderr output."""
    new_text = Signal(str)

    def write(self, text: str) -> None:
        """
        Write text to the stream redirector

        :param text: text to be written
        """
        self.new_text.emit(text)

    def flush(self) -> None:
        """
        A dummy method required only for compatibility with the Python IO system
        """
        # Required for compatibility with Python's IO system
