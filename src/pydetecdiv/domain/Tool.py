"""
Tool module to handle tool definition, requirements and running them in an appropriate environment.
"""
import os
import platform
import re
import subprocess
import xml
import yaml

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

    def set_env_command(self):
        match self.environment['type']:
            case 'conda':
                return self._set_conda_env_command()
            case _:
                print('Unknown environment type')
                return None

    def _set_conda_env_command(self):
        conda_dir = get_config_value('project.conda', 'dir')
        env_name = self.environment["env"]
        if platform.system() == 'Windows':
            conda_exe = os.path.join(conda_dir, 'condabin', 'conda.bat')
            cmd = f'{conda_exe} activate {env_name}'
        else:
            cmd = f'conda run -n {env_name} --cwd $__working_directory__'
        return cmd


class Inputs:
    """
    A class to handle Tool's input as defined in the configuration file
    """

    def __init__(self, element):
        self.element = element
        self.list = {p['name']: Input(p['name'], p['type'], p['format'], label=p['label'] if p['label'] else None)
                     for p in self.parameters}

    @property
    def parameters(self):
        """
        All the parameters that should be passed to the command line
        :return: list of parameters
        :rtype: list
        """
        return [p.attrib for p in self.element.findall('.//param')]


class Input:
    def __init__(self, name, type_, format_, label=None):
        self.name = name
        self.type = type_
        self.format = format_
        self.label = label
        self.value = None

    def is_image(self):
        return self.type == 'data' and self.format in ['imagetiff']


class Outputs:
    """
    A class to handle Tool's input as defined in the configuration file
    """

    def __init__(self, element):
        self.element = element
        self.list = {p['name']: Output(p['name'], p['format'], label=p['label'] if p['label'] else None)
                     for p in self.data}

    @property
    def data(self):
        return [p.attrib for p in self.element.findall('.//data')]


class Output:
    def __init__(self, name, format_, label=None):
        self.name = name
        self.format = format_
        self.label = label
        self.value = None

    def is_image(self):
        return self.format in ['imagetiff']


class Tool:
    """
    A class for handling and running tools specified by XML files
    """

    def __init__(self, path):
        self.path = os.path.dirname(path)
        self.xml_tree = xml.etree.ElementTree.parse(path)
        shed_file = os.path.join(os.path.dirname(path), '.shed.yml')
        with open(shed_file, encoding='utf-8') as file:
            self.shed_content = yaml.load(file, Loader=yaml.FullLoader)
        self.inputs = Inputs(self.root.find('./inputs'))
        self.outputs = Outputs(self.root.find('./outputs'))
        self.requirements = Requirements(self.root.find('./requirements/package'))

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
    def command(self):
        """
        Property returning the command line
        :return:
        """
        return f'{self.requirements.set_env_command()} \'{self.root.find("command").text}\''

    @property
    def attributes(self):
        """
        Convenience property returning the tool's attributes as defined in the root node
        :return: attributes
        :rtype: dict
        """
        return self.root.attrib

    def tests(self):
        for test in self.root.findall('.//test'):
            params = {i.attrib['name']: i.attrib['value'] for i in test.findall('.//param')}
            params.update({o.attrib['name']: o.attrib['file'] for o in test.findall('.//output')})
            yield (params)

    @property
    def parameters(self):
        params = {name: value for name, value in self.inputs.list.items()}
        params.update({name: value for name, value in self.outputs.list.items()})
        return params
