"""
CommandLineTool module to handle tool definition, requirements and running them in an appropriate environment.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydetecdiv.domain.Project import Project

import json
import os
import platform
import re
import subprocess
import xml
from datetime import datetime
from typing import Callable
from xml.etree.ElementTree import Element

import yaml

from pydetecdiv.domain.Dataset import Dataset
from pydetecdiv.domain.tools import Plugins
from pydetecdiv.settings import get_config_value
from pydetecdiv.domain.parameters import ParameterFactory, Parameter


def list_tools() -> dict:
    """
    Provide a list of available tools arranged by categories
    :return: the list of available tools and categories
    :rtype: dict
    """
    toolbox_path = get_config_value('paths', 'toolbox')
    json_data = json.load(open(os.path.join(toolbox_path, 'toolboxes.json'), encoding='utf-8'))
    tool_list = {c['name']: [] for c in json_data['categories']}
    for current_path, _, files in os.walk(os.path.abspath(os.path.join(toolbox_path, 'tools'))):
        for file in files:
            if file.endswith('.xml'):
                tool = CommandLineTool(os.path.join(current_path, file))
                for category in tool.categories:
                    tool_list[category].append(tool)
    return tool_list


class Requirements:
    """
    Class to handle tool requirements, install an environment and the required packages.
    """

    def __init__(self, element: Element) -> None:
        self.element = element

    @property
    def packages(self) -> str:
        """
        Return a list of required packages for the tool

        :return: the list of required packages
        :rtype: list of str
        """
        return self.element.text

    @property
    def environment(self) -> dict:
        """
        Return the environment for the current tool, specifying the type of environment ('conda' or docker')
        and its name

        :return: definition for the tool environment
        :rtype: dict
        """
        return self.element.attrib

    def check_env(self) -> bool:
        """
        Check the environment for this tool already exists

        :param env: the environment to check for existence
        :type env: dict
        :return: True if environment exists, False otherwise
        :rtype: bool
        """
        match self.environment['type']:
            case 'conda':
                return self._check_conda_env()
            case 'plugin':
                return True
            case _:
                print('Unknown environment type')
        return False

    def _check_conda_env(self) -> bool:
        """
        Check whether the required conda environment exists or not

        :return: True if the conda environment exists, False otherwise
        :rtype: bool
        """
        conda_dir = get_config_value('project.conda', 'dir')
        if platform.system() == 'Windows':
            conda_exe = os.path.join(conda_dir, 'condabin', 'conda.bat')
            cmd = f'{conda_exe} env list'
        else:
            conda_exe = os.path.join(conda_dir, 'etc', 'profile.d', 'conda.sh')
            cmd = f'/bin/bash {conda_exe} && conda env list'
        output = subprocess.run(cmd, shell=True, check=True, capture_output=True).stdout.decode('utf-8')
        pattern = re.compile(self.environment['env'], re.MULTILINE)
        if pattern.search(output):
            return True
        return False

    def install(self) -> None:
        """
        Install the requirements if needed
        """
        if not self.check_env():
            self.install_env()

    def install_env(self) -> None:
        """
        Install the environment
        """
        match self.environment['type']:
            case 'conda':
                self._create_conda_env()
            case 'plugin':
                ...
            case _:
                print('Unknown environment type')

    def _create_conda_env(self) -> None:
        """
        Create the required conda environment with the necessary packages
        """
        conda_dir = get_config_value('project.conda', 'dir')
        env_name = self.environment["env"]
        if platform.system() == 'Windows':
            conda_exe = os.path.join(conda_dir, 'condabin', 'conda.bat')
            cmd = f'{conda_exe} create -y -n {env_name} {self.packages}'
        else:
            conda_exe = os.path.join(conda_dir, 'etc', 'profile.d', 'conda.sh')
            cmd = f'/bin/bash {conda_exe} && conda create -y -n {env_name} {self.packages}'
        subprocess.run(cmd, shell=True, check=True, )


class Inputs:
    """
    A class to handle CommandLineTool's input as defined in the configuration file
    """

    def __init__(self, tool: 'CommandLineTool'):
        self.element: Element = tool.root.find('./inputs')
        self.list: dict[str, Parameter] = {p.attrib['name']: ParameterFactory().create(p, tool) for p in
                                           self.element.findall('.//param')}

    @property
    def values(self) -> list[Parameter]:
        """
        Convenience property returning the values of all input parameters in the self.list

        :return: the values of all input parameters in the self.list
        :rtype: list
        """
        return list(self.list.values())


class Outputs:
    """
    A class to handle CommandLineTool's input as defined in the configuration file
    """

    def __init__(self, tool: 'CommandLineTool') -> None:
        self.element = tool.root.find('./outputs')
        self.list = {p.attrib['name']: ParameterFactory().create(p, tool) for p in self.element.findall('.//data')}


class Command:
    """
    A class handling commands for running tools. A command can be a command-line or a call to the execute method of a
    class inheriting from CommandLineTool (i.e. generic tool) and representing a particular tool
    """

    def __init__(self, command: str, requirements: Requirements, tool_path: str):
        self.code = command
        if tool_path:
            self.code = self.code.replace('$__tool_directory__', os.path.join(tool_path, ''))
        self.requirements = requirements
        self.working_dir = '.'
        self.parameters = {}
        self.dataset = None

    def set_env_command(self) -> str | Callable | None:
        """
        Return the command to set up the environment required to run the tool

        :return: command to set the environment up
        :rtype: str
        """
        match self.requirements.environment['type']:
            case 'conda':
                return self._set_conda_env_command()
            case 'plugin':
                return self.go_to_working_dir
            case _:
                print('Unknown environment type')
                return None

    def _set_conda_env_command(self) -> str:
        """
        Return the command to set up the conda environment required to run the tool

        :return: command to set the conda environment up
        :rtype: str
        """
        conda_dir = get_config_value('project.conda', 'dir')
        env_name = self.requirements.environment["env"]
        if platform.system() == 'Windows':
            conda_exe = os.path.join(conda_dir, 'condabin', 'conda.bat')
            cmd = f'{conda_exe} activate {env_name}'
        else:
            cmd = f'conda run -n {env_name} --cwd {self.working_dir}'
        return cmd

    def set_dataset(self, dataset: Dataset) -> 'Command':
        self.dataset = dataset
        return self

    def set_working_dir(self, working_dir):
        """
        Set the working directory

        :param working_dir: the working directory path
        :type: os.Path
        :return: the command object
        :rtype: Command
        """
        self.working_dir = working_dir
        return self

    def go_to_working_dir(self):
        """
        Go to the working directory.
        """
        os.chdir(self.working_dir)

    def set_parameters(self, parameters: dict[str, Parameter]) -> 'Command':
        """
        Set the parameters for running the command

        :param parameters: the parameters
        :type parameters: dict
        :return: the command object
        :rtype: Command
        """
        self.parameters = parameters
        return self

    def execute(self) -> dict[str, str]:
        """
        Execute the command

        :return: the output of the job
        :rtype: subprocess.CompletedProcess
        """
        if self.requirements.environment['type'] == 'plugin':
            plugin = Plugins(self.requirements.packages).list[self.code.strip()]
            output = plugin(self.parameters, self.dataset, self.working_dir).run()
        else:
            for name, param in self.parameters.items():
                self.code = self.code.replace('${' + name + '}', param.value)
            command = f'{self.set_env_command()} \'{self.code}\''
            output = subprocess.run(command, shell=True, check=True, capture_output=True)
            output = {'stdout': output.stdout.decode('utf-8'), 'stderr': output.stderr.decode('utf-8')}
        return output


class CommandLineTool:
    """
    A class for handling tools specified by XML files. A CommandLineTool object represents a generic tool in the toolbox. Internal
    tools must inherit from this class with the addition of a supplementary method implementing the tool's algorithm.
    """

    def __init__(self, path: str):
        self.path = os.path.dirname(path) if path else None
        self.xml_tree = xml.etree.ElementTree.parse(path)
        shed_file = os.path.join(os.path.dirname(path), '.shed.yml')
        with open(shed_file, encoding='utf-8') as file:
            self.shed_content = yaml.load(file, Loader=yaml.FullLoader)
        self.inputs = Inputs(self)
        self.outputs = Outputs(self)
        self.requirements = Requirements(self.root.find('./requirements/package'))
        self.command = Command(self.root.find("command").text, self.requirements, self.path)

    def init_dso_inputs(self, project: 'Project' = None) -> None:
        """
        Initialize the DSOs for input parameters and place them in the dso field.

        :param project:
        """
        for i in self.inputs.values:
            i.set_dso(project)

    def init_test(self, test_param: dict[str, str], project: 'Project' = None):
        """
        Initialize values of parameters for testing purposes. If an input defines a DSO then its dso field is set to the
        corresponding DSO or list thereof.

        :param test_param: the parameters for the test
        :type test_param: dict
        :param project: the pyDetecDiv project the test must be run on
        :type project: Project
        """
        for name, value in test_param.items():
            if name in self.parameters:
                if self.parameters[name].is_multiple():
                    self.parameters[name].value = value.split(',')
                else:
                    self.parameters[name].value = value
        self.init_dso_inputs(project=project)

    @property
    def root(self) -> Element:
        """
        Convenience property returning the root of the CommandLineTool's XML configuration

        :return: the root of the XML tool definition
        :rtype: xml.etree.Element 'tool'
        """
        return self.xml_tree.getroot()

    @property
    def name(self) -> str:
        """
        Property returning the name of the tool

        :return: tool name
        :rtype: str
        """
        return self.root.get('name')

    @property
    def version(self) -> str:
        """
        Property returning the version of the tool

        :return: tool version
        :rtype: str
        """
        return self.root.get('version')

    @property
    def categories(self) -> list[str]:
        """
        Property returning the categories the tool belongs to

        :return: tool categories
        :rtype: list of str
        """
        return self.shed_content["categories"]

    @property
    def command_line(self) -> str:
        """
        Property returning the command line

        :return:
        """
        return f'{self.command.set_env_command()} \'{self.command.code}\''

    @property
    def attributes(self) -> dict:
        """
        Convenience property returning the tool's attributes as defined in the root node

        :return: attributes
        :rtype: dict
        """
        return self.root.attrib

    def tests(self) -> str:
        """
        Return the parameters for running the testing
        """
        for test in self.root.findall('.//test'):
            params = {i.attrib['name']: i.attrib['value'] for i in test.findall('.//param')}
            params.update({o.attrib['name']: o.attrib['file'] for o in test.findall('.//output')})
            yield params

    @property
    def parameters(self) -> dict[str, Parameter]:
        """
        Return the all the input and output parameters for running the tool

        :return: input and ouput parameters
        :rtype: dict of Parameter objects (Input and Output)
        """
        params = self.inputs.list.copy()
        params.update(self.outputs.list)
        return params

    @property
    def dataset(self) -> str:
        if 'dataset' in self.parameters:
            return self.parameters['dataset'].value
        dataset_name = f'{self.name}_{datetime.now()}'
        return dataset_name.replace(' ', '_')
