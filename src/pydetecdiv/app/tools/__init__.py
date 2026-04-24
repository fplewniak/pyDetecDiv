"""
Abstract Tool class
"""
import datetime
import os
from abc import ABC
from os import makedirs

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
        self._log_text = ''
        self.run = None
        PyDetecDiv.app.project_selected.connect(self.project_selected)

    def project_selected(self):
        """
        Create the base working directory for the tool when a new project is selected
        """
        os.makedirs(self.working_dir, exist_ok=True)

    @property
    def working_dir(self)-> str:
        """
        The working directory for the tool

        :return: the working directory
        """
        if get_project_dir() is not None:
            return os.path.join(get_project_dir(), self._working_dir)
        return None

    def run_path(self, run: Run = None) -> str:
        """
        Return the path for the specified run where to save related results and log files which do not fit in repository

        :param run: the run
        :return: the path
        """
        path = os.path.join(self.working_dir, 'runs', str(run.id_))
        makedirs(path, exist_ok=True)
        return path

    def log_path(self, run: Run = None) -> str:
        """
        The path of the log file

        :param run: the corresponding run
        :return: the path of the log file
        """
        path = os.path.join(self.run_path(run), 'log.txt')
        return path

    def log_text(self, text: str) -> None:
        """
        Sends standard output text to log file

        :param text: the captured stdout text
        """
        self._log_text = self._log_text + text
        if self.run is not None:
            with open(self.log_path(self.run), 'a', encoding='utf-8') as f:
                f.write(self._log_text)
                self._log_text = ''

    def save_run(self, command: str, param_list: list[Parameter] = None, key_val: dict = None):
        """
        Saves the run for this tool
        """
        if param_list is None:
            param_list = []

        param_list.extend([p for p in self.parameters.parameter_list if p.should_be_saved])

        for parameter in param_list:
            parameter.should_be_saved = False

        if key_val is None:
            key_val = {}
        key_val.update({'date': datetime.datetime.now().isoformat()})

        record = {
            'tool_name'   : self.id_,
            'tool_version': self.version,
            'is_plugin'   : False,
            'command'     : command,
            'parameters'  : self.parameters.json(param_list=param_list),
            'key_val'     : key_val,
            # 'uuid': self.uuid
            }
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.run = Run(project=project, **record)
        # project.commit()
        return self.run

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
