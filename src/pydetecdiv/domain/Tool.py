"""

"""
import os
import re
import xml

import numpy as np
import yaml

class Requirements:
    def __init__(self, element):
        self.element = element

    @property
    def packages(self):
        return set(np.array([str.split(e.text, ' ') for e in self.element]).flatten())

    @property
    def environments(self):
        return [e.attrib for e in self.element]

    def install(self):
        for env, packages in [(e.attrib, e.text) for e in self.element]:
            if not self.check_env(env):
                self.install_env(env)
            self.install_packages(env, packages)

    def check_env(self, env):
        match env['type']:
            case 'conda':
                print(f'conda env list --json')
        return True

    def install_env(self, env):
        match env['type']:
            case 'conda':
                print(f'conda create --name {env["env"]}')

    def install_packages(self, env, packages):
        match env['type']:
            case 'conda':
                print(f'conda activate {env["env"]}')
                print(f'conda install {packages}')

class Inputs:
    def __init__(self, element):
        self.element = element

    @property
    def parameters(self):
        return [p.attrib for p in self.element.findall('param')]

class Tool:
    def __init__(self, path):
        self.path = os.path.dirname(path)
        self.xml_tree = xml.etree.ElementTree.parse(path)
        shed_file = os.path.join(os.path.dirname(path), '.shed.yml')
        with open(shed_file) as file:
            self.shed_content = yaml.load(file, Loader=yaml.FullLoader)

    @property
    def root(self):
        return self.xml_tree.getroot()

    @property
    def name(self):
        return self.root.get('name')

    @property
    def version(self):
        return self.root.get('version')

    @property
    def categories(self):
        return self.shed_content["categories"]

    @property
    def requirements(self):
        return Requirements(self.root.findall('./requirements/package'))

    @property
    def command(self):
        return re.sub(r'\$__tool_directory__', os.path.join(self.path, ''), self.root.find('command').text)

    @property
    def attributes(self):
        return self.root.attrib

    @property
    def inputs(self):
        return Inputs(self.root.find('./inputs'))

    def run(self):
        self.requirements.install()
        for command in str.splitlines(self.command):
            print(command.strip())
        print(self.inputs.parameters)

