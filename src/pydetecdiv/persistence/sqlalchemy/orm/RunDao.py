#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from sqlalchemy import Column, Integer, String
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class RunDao(DAO, Base):
    """
    DAO class for access to BioImageIT dataset records from the SQL database
    """
    __tablename__ = 'run'
    exclude = ['id_']
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    uuid = Column(String(36),)
    process_name = Column(String, nullable=False,)
    process_url = Column(String, nullable=False,)
    inputs = Column(String)
    parameters = Column(String)

    @property
    def record(self):
        """
        A method creating a record dictionary from a dataset row dictionary. This method is used to convert the SQL
        table columns into the dataset record fields expected by the domain layer
        :return a dataset record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'uuid': self.uuid,
                'process_name': self.process_name,
                'process_url': self.process_url,
                'inputs': self.inputs,
                'parameters': self.parameters
                }