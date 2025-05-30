#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to Dataset data
"""
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.types import JSON
from sqlalchemy.orm import relationship, joinedload
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class DatasetDao(DAO, Base):
    """
    DAO class for access to dataset records from the SQL database
    """
    __tablename__ = 'dataset'
    exclude = ['id_']
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    uuid = Column(String(36), )
    name = Column(String, unique=True, nullable=False)
    url = Column(String)
    type_ = Column(String)
    run = Column(Integer, ForeignKey('run.id_'), nullable=True, index=True)
    pattern = Column(String)
    key_val = Column(JSON)

    data_list_ = relationship('DataDao')

    def data_list(self, dataset_id: int) -> list[dict[str, object]]:
        """
        A method returning the list of Data records whose parent Dataset has id\_ == dataset_id

        :param dataset_id: the id of the Dataset
        :return: a list of Data records whose parent Dataset has id\_ == dataset_id
        """
        if self.session.query(DatasetDao).filter(DatasetDao.id_ == dataset_id).first() is not None:
            data_list = [data.record
                         for data in self.session.query(DatasetDao)
                         .options(joinedload(DatasetDao.data_list_))
                         .filter(DatasetDao.id_ == dataset_id)
                         .first().data_list_]
        else:
            data_list = []
        return data_list

    @property
    def record(self) -> dict[str, object]:
        """
        A method creating a record dictionary from a dataset row dictionary. This method is used to convert the SQL
        table columns into the dataset record fields expected by the domain layer

        :return: a dataset record as a dictionary with keys() appropriate for handling by the domain layer
        """
        return {'id_': self.id_,
                'uuid': self.uuid,
                'name': self.name,
                'url': self.url,
                'type_': self.type_,
                'run': self.run,
                'pattern': self.pattern,
                'key_val': self.key_val,
                }
