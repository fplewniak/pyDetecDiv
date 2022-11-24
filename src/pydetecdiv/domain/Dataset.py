#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from pydetecdiv.domain.dso import NamedDSO


class Dataset(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of data
    """

    def __init__(self, uuid, url, type_, run, **kwargs):
        super().__init__(**kwargs)
        self.uuid = uuid
        self.url = url
        self.type_ = type_
        self.run = run
        self.validate()

    def check_validity(self):
        """
        Checks the current Data object is valid and consistent with its image content. If
        not, the shape values are updated
        """
        ...

    def validate(self, updated=True):
        """
        Validates the current Data
        :param updated: True if the Data has been updated, False otherwise
        :type updated: bool
        """
        ...

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
        Returns a record dictionary of the current Data
        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to
        another
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'name': self.name,
            'url': self.url,
            'type_': self.type_,
            'run': self.run,
            'uuid': self.uuid
        }
        if not no_id:
            record['id_'] = self.id_
        return record
