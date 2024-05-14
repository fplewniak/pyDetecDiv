#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
import os

from pydetecdiv.domain.dso import NamedDSO
from pydetecdiv.settings import get_config_value


class Dataset(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of data
    """

    def __init__(self, url='', type_=None, run=None, pattern=None, key_val=None, **kwargs):
        super().__init__(**kwargs)
        self.url_ = url
        self.type_ = type_
        self.run = run
        self.pattern = pattern
        self.key_val = key_val
        self.validate(updated=False)

    @property
    def url(self):
        """
        URL property of the data file, relative to the workspace directory or absolute path if file are stored in place

        :return: relative or absolute path of the data file
        :rtype: str
        """
        if os.path.isabs(self.url_):
            return self.url_
        return os.path.join(get_config_value('project', 'workspace'), self.project.dbname, self.name)

    @property
    def data_list(self):
        """
        returns the list of data files in dataset

        :return: list of Data objects
        :rtype: list of Data objects
        """
        return self.project.get_linked_objects('Data', to=self)

    def record(self, no_id=False):
        """
        Returns a record dictionary of the current Dataset

        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to another
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'name': self.name,
            'url': self.url,
            'type_': self.type_,
            'run': self.run,
            'pattern': self.pattern,
            'uuid': self.uuid,
            'key_val': self.key_val,
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    @property
    def info(self):
        return f"""
   Name: {self.name}
   Path: {self.url}
   Type: {self.type_}
    Run: {self.run}
Pattern: {self.pattern}
        """
