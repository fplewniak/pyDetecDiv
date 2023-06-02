"""
Tool module to handle tool definition, requirements and running them in an appropriate environment.
"""
import importlib
import inspect
import os
import pkgutil
import platform
import re
import subprocess
import xml
import yaml

import pydetecdiv
from pydetecdiv.domain.tools import Plugins
from pydetecdiv.utils import remove_keys_from_dict
from pydetecdiv.settings import get_config_value


class Requirements:
    """
    Class to handle tool requirements, install an environment and the required packages.
    """

    def __init__(self, element):
        self.element = element

    @property
    def packages(self):
        """
        Return a list of required packages for the tool
        :return: the list of required packages
        :rtype: list of str
        """
        return self.element.text
        # return set(str.split(self.element.text, ' '))

    @property
    def environment(self):
        """
        Return the environment for the current tool, specifying the type of environment ('conda' or docker')
        and its name
        :return: definition for the tool environment
        :rtype: dict
        """
        return self.element.attrib

    def check_env(self):
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

    def _check_conda_env(self):
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

    def install(self):
        """
        Install the requirements if needed
        """
        if not self.check_env():
            self.install_env()

    def install_env(self):
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

    def _create_conda_env(self):
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


class Parameter:
    """
    A generic parameter class to represent both inputs and outputs parameters
    """

    def __init__(self, name, type_, **kwargs):
        self.name = name
        self.type = type_
        self.format = kwargs['format'] if type_ == 'data' and 'format' in kwargs else None
        self.label = kwargs['label'] if 'label' in kwargs else None
        self.value = None
        self.obj = None

    def is_image(self):
        """
        Check the parameter represents image data
        :return: True if the parameter represents image data, False otherwise
        :rtype: bool
        """
        return self.type == 'data' and self.format in ['imagetiff']


class Inputs:
    """
    A class to handle Tool's input as defined in the configuration file
    """

    def __init__(self, element):
        self.element = element
        self.list = {p['name']: Input(p['name'], p['type'], **remove_keys_from_dict(p, ['name', 'type']))
                     for p in [p.attrib for p in self.element.findall('.//param')]}

    @property
    def values(self):
        return self.list.values()


class Input(Parameter):
    """
    A class representing an input parameter
    """


class Outputs:
    """
    A class to handle Tool's input as defined in the configuration file
    """

    def __init__(self, element):
        self.element = element
        self.list = {p['name']: Output(p['name'], p['format'], **remove_keys_from_dict(p, ['name', 'format']))
                     for p in [p.attrib for p in self.element.findall('.//data')]}


class Output(Parameter):
    """
    A class representing an output parameter
    """

    def __init__(self, name, format_, **kwargs):
        super().__init__(name, 'data', format=format_, **kwargs)

    def is_image(self):
        """
        Check the output parameter represents image data
        :return: True if the output parameter represents image data, False otherwise
        :rtype: bool
        """
        return self.format in ['imagetiff']


class Command:
    """
    A class handling commands for running tools. A command can be a command-line or a call to the execute method of a
    class inheriting from Tool (i.e. generic tool) and representing a particular tool
    """

    def __init__(self, command, requirements, tool_path):
        self.code = command
        if tool_path:
            self.code = self.code.replace('$__tool_directory__', os.path.join(tool_path, ''))
        self.requirements = requirements
        self.working_dir = '.'
        self.parameters = {}

    def set_env_command(self):
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

    def _set_conda_env_command(self):
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

    def set_parameters(self, parameters):
        """
        Set the parameters for running the command
        :param parameters: the parameters
        :type parameters: dict
        :return: the command object
        :rtype: Command
        """
        # self.parameters = {name: param.value for name, param in parameters.items()}
        self.parameters = {}
        for name, param in parameters.items():
            self.parameters[name] = param.value if param.obj is None else param.obj
        return self

    def execute(self):
        """
        Execute the command
        :return: the output of the job
        :rtype: subprocess.CompletedProcess
        """
        if self.requirements.environment['type'] == 'plugin':
            plugin = Plugins(self.requirements.packages).list[self.code.strip()]
            return plugin(self.parameters).run()
        else:
            for name, param in self.parameters.items():
                self.code = self.code.replace('${' + name + '}', param)
            command = f'{self.set_env_command()} \'{self.code}\''
            output = subprocess.run(command, shell=True, check=True, capture_output=True)
            return {'stdout': output.stdout.decode('utf-8'), 'stderr': output.stderr.decode('utf-8')}


class Tool:
    """
    A class for handling tools specified by XML files. A Tool object represents a generic tool in the toolbox. Internal
    tools must inherit from this class with the addition of a supplementary method implementing the tool's algorithm.
    """

    def __init__(self, path):
        self.path = os.path.dirname(path) if path else None
        self.xml_tree = xml.etree.ElementTree.parse(path)
        shed_file = os.path.join(os.path.dirname(path), '.shed.yml')
        with open(shed_file, encoding='utf-8') as file:
            self.shed_content = yaml.load(file, Loader=yaml.FullLoader)
        self.inputs = Inputs(self.root.find('./inputs'))
        self.outputs = Outputs(self.root.find('./outputs'))
        self.requirements = Requirements(self.root.find('./requirements/package'))
        self.command = Command(self.root.find("command").text, self.requirements, self.path)

    def init_dso_inputs(self, project=None):
        for i in self.inputs.values:
            if i.format in ['FOV', 'ROI', 'Data', 'Dataset']:
                i.obj = project.get_named_object(i.format, i.value)

    def init_test(self, test_param, project=None):
        for name, value in test_param.items():
                if name in self.parameters:
                    self.parameters[name].value = value
        self.init_dso_inputs(project=project)

    @property
    def root(self):
        """
        Convenience property returning the root of the Tool's XML configuration
        :return: the root of the XML tool definition
        :rtype: xml.etree.Element 'tool'
        """
        return self.xml_tree.getroot()

    @property
    def name(self):
        """
        Property returning the name of the tool
        :return: tool name
        :rtype: str
        """
        return self.root.get('name')

    @property
    def version(self):
        """
        Property returning the version of the tool
        :return: tool version
        :rtype: str
        """
        return self.root.get('version')

    @property
    def categories(self):
        """
        Property returning the categories the tool belongs to
        :return: tool categories
        :rtype: list of str
        """
        return self.shed_content["categories"]

    @property
    def command_line(self):
        """
        Property returning the command line
        :return:
        """
        return f'{self.command.set_env_command()} \'{self.command.code}\''

    @property
    def attributes(self):
        """
        Convenience property returning the tool's attributes as defined in the root node
        :return: attributes
        :rtype: dict
        """
        return self.root.attrib

    def tests(self):
        """
        Return the parameters for running the testing
        """
        for test in self.root.findall('.//test'):
            params = {i.attrib['name']: i.attrib['value'] for i in test.findall('.//param')}
            params.update({o.attrib['name']: o.attrib['file'] for o in test.findall('.//output')})
            yield params

    @property
    def parameters(self):
        """
        Return the all the input and output parameters for running the tool
        :return: input and ouput parameters
        :rtype: dict of Parameter objects (Input and Output)
        """
        params = self.inputs.list.copy()
        params.update(self.outputs.list)
        return params
