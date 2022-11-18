#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from sqlalchemy import Column, Integer, String, ForeignKey, text, Date
from sqlalchemy.orm import joinedload, relationship
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base
import pydetecdiv.persistence.sqlalchemy.orm.dao as dao


class ExperimentDao(DAO, Base):
    """
    DAO class for access to BioImageIT data records from the SQL database
    """
    __tablename__ = 'experiment'

    uuid = Column(String(36), primary_key=True)
    name = Column(String, unique=True, nullable=False)
    author = Column(String)
    date = Column(Date)
    raw_dataset = Column(String, ForeignKey('dataset.uuid'), index=True)

    @property
    def record(self):
        """
        A method creating a record dictionary from a data row dictionary. This method is used to convert the SQL
        table columns into the data record fields expected by the domain layer
        :return a data record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'uuid': self.uuid,
                'name': self.name,
                'author': self.author,
                'date': self.date,
                'raw_dataset': self.raw_dataset,
                }
