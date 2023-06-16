#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
import json
import os

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
            'command': self.tool.command_line,
            'parameters': json.dumps({name: param.value for name, param in self.tool.parameters.items()}),
            'uuid': self.uuid
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    def execute(self, testing=False):
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
        self.tool.command.set_dataset(self.tool.dataset)
        output = self.tool.command.set_working_dir(working_dir).set_parameters(self.tool.parameters).execute()
        print(output['stdout'])
        print(output['stderr'])

    def test(self):
        """
        Run the testing
        """
        for t in self.tool.tests():
            self.tool.init_test(t, project=self.project)
            self.execute(testing=True)
