#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from sqlalchemy import Column, String, Integer, ForeignKey, Date
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class ExperimentDao(DAO, Base):
    """
    DAO class for access to BioImageIT data records from the SQL database
    """
    __tablename__ = 'experiment'
    exclude = ['id_']
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    uuid = Column(String(36),)
    name = Column(String, unique=True, nullable=False)
    author = Column(String)
    date = Column(Date)
    raw_dataset = Column(String, ForeignKey('dataset.id_'), index=True)

    @property
    def record(self):
        """
        A method creating a record dictionary from a data row dictionary. This method is used to convert the SQL
        table columns into the data record fields expected by the domain layer

        :return: a data record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'uuid': self.uuid,
                'name': self.name,
                'author': self.author,
                'date': self.date,
                'raw_dataset': self.raw_dataset,
                }
