#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
import json
import os
import subprocess

from pydetecdiv.domain.dso import DomainSpecificObject


class Run(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of data
    """

    def __init__(self, tool, **kwargs):
        super().__init__(**kwargs)
        self.tool = tool
        self.validate()

    def record(self, no_id=False):
        """
        Returns a record dictionary of the current Data
        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to
        another
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'tool_name': self.tool.name,
            'tool_version': self.tool.version,
            'command': self.tool.command,
            'parameters': json.dumps(self.tool.inputs.parameters),
            'uuid': self.uuid
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    def execute(self):
        """
        Execute the job after having installed requirements if necessary
        """
        self.tool.requirements.install()
        for t in self.tool.tests():
            command = self.tool.command.replace('$__tool_directory__', os.path.join(self.tool.path, ''))
            for name, value in t.items():
                command = command.replace('${' + name + '}', value)
            command = f'cd {self.tool.path}/test-data\n{self.tool.requirements.set_env_command()}\n{command}'
            output = subprocess.run(command, shell=True, check=True, capture_output=True)
            print(output.stdout.decode('utf-8'))
            print(output.stderr.decode('utf-8'))
