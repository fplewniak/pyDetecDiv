#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A module defining the business logic classes and methods that can be applied to Experiment
"""
import datetime

from pydetecdiv.domain.Dataset import Dataset
from pydetecdiv.domain.dso import NamedDSO


class Experiment(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of an experiment
    """

    def __init__(self, uuid: str, author: str, date: datetime.datetime, raw_dataset: int, **kwargs):
        super().__init__(**kwargs)
        self.uuid = uuid
        self.author = author
        self.date = date
        self.raw_dataset_ = raw_dataset
        self.validate(updated=False)

    @property
    def raw_dataset(self) -> Dataset:
        """
        property returning the raw dataset object for this experiment

        :return: raw dataset
        """
        return self.project.get_object('Dataset', id_=self.raw_dataset_)

    def record(self, no_id: bool = False) -> dict:
        """
        Returns a record dictionary of the current Experiment object

        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to another
        :return: record dictionary
        """
        record = {
            'name'       : self.name,
            'author'     : self.author,
            'date'       : self.date,
            'raw_dataset': self.raw_dataset_,
            'uuid'       : self.uuid
            }
        if not no_id:
            record['id_'] = self.id_
        return record
