"""
Abstract Tool class
"""
import os
from abc import abstractmethod, ABC

from pydetecdiv.app import get_project_dir, PyDetecDiv
from pydetecdiv.app.parameters import Parameters


class Tool(ABC):
    """
    Abstract Tool class providing the basic functionalities for the tool
    """
    id_ = None
    version = '1.0.0'
    name = None

    def __init__(self, parameters: Parameters | None = None, working_dir: str | None = None):
        self.parameters = parameters
        self._working_dir = working_dir
        self._command: str | None = None
        PyDetecDiv.app.project_selected.connect(self.project_selected)

    def project_selected(self):
        os.makedirs(self.working_dir, exist_ok=True)

    @property
    def working_dir(self):
        if get_project_dir() is not None:
            return os.path.join(get_project_dir(), self._working_dir)
        return None

    @abstractmethod
    def save_run(self, *args, **kwargs):
        """
        Saves the run for this tool
        """

    @property
    def command(self) -> str:
        """
        The command or method used to run the tool

        :return: the command or method name
        """
        if self._command is None:
            return self.id_
        return self._command

    @command.setter
    def command(self, command: str) -> None:
        self._command = command
