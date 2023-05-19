"""
Tool module to handle tool definition, requirements and running them in an appropriate environment.
"""
import os
import re
import xml
import yaml


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
        return set(str.split(self.element.text, ' '))

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
                print('conda env list --json')
                return False
        return True

    def check_package(self, package):
        """
        Check the package are already installed in the environment
        :param package: the package to check installation for
        :type package: str
        :return: True if the package is installed, False otherwise
        :rtype: bool
        """
        match self.environment['type']:
            case 'conda':
                print(f'conda list -n {self.environment["env"]} {package}')
                return False
        return True

    def install(self):
        """
        Install the requirements if needed
        """
        if not self.check_env():
            self.install_env()
        self.install_packages()

    def install_env(self):
        """
        Install the environment
        """
        match self.environment['type']:
            case 'conda':
                print(f'conda create --name {self.environment["env"]}')

    def install_packages(self):
        """
        Install packages in environment
        """
        match self.environment['type']:
            case 'conda':
                print(f'conda activate {self.environment["env"]}')
                packages = [package for package in self.packages if not self.check_package(package)]
                print(f'conda install {" ".join(packages)}')


class Inputs:
    """
    A class to handle Tool's input as defined in the configuration file
    """

    def __init__(self, element):
        self.element = element

    @property
    def parameters(self):
        """
        All the parameters that should be passed to the command line
        :return: list of parameters
        :rtype: list
        """
        return [p.attrib for p in self.element.findall('param')]


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
    def requirements(self):
        """
        Property returning the requirements for this tool
        :return: the environment and packages requirement to run the tool
        :rtype: Requirements object
        """
        return Requirements(self.root.find('./requirements/package'))

    @property
    def command(self):
        """
        Property returning the command line
        :return:
        """
        return re.sub(r'\$__tool_directory__', os.path.join(self.path, ''), self.root.find('command').text)

    @property
    def attributes(self):
        """
        Convenience property returning the tool's attributes as defined in the root node
        :return: attributes
        :rtype: dict
        """
        return self.root.attrib

    @property
    def inputs(self):
        """
        Property providing the defined inputs for the tool
        :return: input parameters to run the tool
        :rtype: Inputs object
        """
        return Inputs(self.root.find('./inputs'))

    def run(self):
        """
        Install the requirements if needed and run the tool
        """
        self.requirements.install()
        for command in str.splitlines(self.command):
            print(command.strip())
        print(self.inputs.parameters)
