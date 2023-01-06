#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from pydetecdiv.domain.dso import NamedDSO


class Experiment(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of data
    """

    def __init__(self, uuid, author, date, raw_dataset, **kwargs):
        super().__init__(**kwargs)
        self.uuid = uuid
        self.author = author
        self.date = date
        self.raw_dataset_ = raw_dataset
        self.validate()

    @property
    def raw_dataset(self):
        """
        property returning the raw dataset object for this experiment
        :return: raw dataset
        :rtype: Dataset object
        """
        return self.project.get_object('Dataset', id_=self.raw_dataset_)

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
            'author': self.author,
            'date': self.date,
            'raw_dataset': self.raw_dataset_,
            'uuid': self.uuid
        }
        if not no_id:
            record['id_'] = self.id_
        return record
