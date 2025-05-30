#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
import json
import os
from datetime import datetime
from typing import Any

from pydetecdiv import generate_uuid
from pydetecdiv.domain.CommandLineTool import CommandLineTool
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.settings import get_config_value


class Run(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of data
    """

    def __init__(self, tool_name: str = None, tool_version: str = None, is_plugin: bool = False, command: str = None,
                 parameters:  dict[str, object] = None, key_val: dict[str, Any] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.is_plugin = is_plugin
        self.command = command
        self.parameters = parameters
        self.key_val = key_val
        self.validate()

    def record(self, no_id: bool = False) -> dict[str, Any]:
        """
        Returns a record dictionary of the current Data

        :param no_id: if True, the id\_ is not passed included in the record to allow transfer from one project to another
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'tool_name': self.tool_name,
            'tool_version': self.tool_version,
            'is_plugin': self.is_plugin,
            'command': self.command,
            # 'parameters': json.dumps({name: param.value for name, param in self.parameters.items()}),
            'parameters': self.parameters,
            'uuid': self.uuid,
            'key_val': self.key_val,
        }
        if not no_id:
            record['id_'] = self.id_
        return record

# The following code is not working currently, and will need to be adapted to include command line tools integration
    def execute(self, tool: CommandLineTool, testing: bool = False):
        """
        Execute the job after having installed requirements if necessary
        """
        self.tool.requirements.install()
        if testing:
            working_dir = os.path.join(self.tool.path, 'test-data')
        else:
            working_dir = os.path.join(self.project.path, self.tool.dataset)
            if not os.path.exists(working_dir):
                os.mkdir(working_dir)

        dataset = self.project.get_object('Dataset', self.project.save_record('Dataset', {
            'id_': None,
            'uuid': generate_uuid(),
            'name': self.tool.dataset,
            'url': self.tool.dataset,
            'type_': 'run',
            'run': self.uuid,
            'pattern': None,
        }))
        self.tool.command.set_dataset(self.tool.dataset)
        output = self.tool.command.set_working_dir(working_dir).set_parameters(self.tool.parameters).execute()
        print(self.project.get_linked_objects('Data', dataset))
        if self.project.count_links('Data', dataset) == 0:
            for dirpath, dirnames, filenames in os.walk(working_dir):
                for filename in filenames:
                    print(filename)
                    record = {
                        'id_': None,
                        'uuid': generate_uuid(),
                        'name': filename,
                        'dataset': dataset.uuid,
                        'author': get_config_value('project', 'user'),
                        'date': datetime.now(),
                        'url': filename,
                        'format': self.format_sniffer(filename),
                        'source_dir': '',
                        'meta_data': '{}',
                        'key_val': '{}',
                    }
                    self.project.get_object('Data', self.project.save_record('Data', record))
        print(output['stdout'])
        print(output['stderr'])

    def test(self):
        """
        Run the testing
        """
        for t in self.tool.tests():
            self.tool.init_test(t, project=self.project)
            self.execute(testing=True)

    def format_sniffer(self, filename):
        formats = {
            'tiff': 'imagetiff',
            'tif': 'imagetiff',
            'txt': 'text',
            'csv': 'tabular',
        }
        return os.path.splitext(filename)[1]
