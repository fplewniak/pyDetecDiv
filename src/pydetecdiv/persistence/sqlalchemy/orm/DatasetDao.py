#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from sqlalchemy import Column, Integer, String, ForeignKey, text, Date
from sqlalchemy.orm import joinedload, relationship
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base
import pydetecdiv.persistence.sqlalchemy.orm.dao as dao


class DatasetDao(DAO, Base):
    """
    DAO class for access to BioImageIT dataset records from the SQL database
    """
    __tablename__ = 'dataset'

    uuid = Column(String(36), primary_key=True)
    name = Column(String, unique=True, nullable=False)
    type_ = Column(String)
    run = Column(String, ForeignKey('run.uuid'), nullable=True, index=True)

    @property
    def record(self):
        """
        A method creating a record dictionary from a dataset row dictionary. This method is used to convert the SQL
        table columns into the dataset record fields expected by the domain layer
        :return a dataset record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'uuid': self.uuid,
                'name': self.name,
                'type': self.type_,
                'run': self.run
                }
