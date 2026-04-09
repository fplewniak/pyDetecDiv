"""
Abstract Tool class
"""
import datetime
import os
from abc import abstractmethod, ABC

from pydetecdiv.app import get_project_dir, PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.parameters import Parameters, Parameter
from pydetecdiv.domain.Run import Run


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

    def save_run(self, command: str, param_list: list[Parameter] = None, groups: list[str] | str = None):
        """
        Saves the run for this tool
        """
        if groups is None and param_list is None:
            groups = command

        record = {
            'tool_name'   : self.id_,
            'tool_version': self.version,
            'is_plugin'   : False,
            'command'     : command,
            'parameters'  : self.parameters.json(groups=groups, param_list=param_list),
            'key_val'     : {'date': datetime.datetime.now().isoformat()}
            # 'uuid': self.uuid
            }
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            run = Run(project=project, **record)
        # project.commit()
        return run

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
