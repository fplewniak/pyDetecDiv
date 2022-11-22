#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from sqlalchemy import Column, Integer, String, ForeignKey, Date
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base
from pydetecdiv.persistence.sqlalchemy.orm.associations import FovData


class DataDao(DAO, Base):
    """
    DAO class for access to BioImageIT data records from the SQL database
    """
    __tablename__ = 'data'

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    uuid = Column(String(36))
    name = Column(String, unique=True, nullable=False)
    dataset = Column(String, ForeignKey('dataset.uuid'), nullable=False, index=True)
    author = Column(String)
    date = Column(Date)
    url = Column(String)
    format = Column(String)
    source_dir = Column(String)
    meta_data = Column(String)
    key_val = Column(String)
    # fov = Column(Integer, ForeignKey('FOV.id_'), index=True)

    fov_list_ = FovData.data_to_fov()

    @property
    def record(self):
        """
        A method creating a record dictionary from a data row dictionary. This method is used to convert the SQL
        table columns into the data record fields expected by the domain layer
        :return a data record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'uuid': self.uuid,
                'name': self.name,
                'dataset': self.dataset,
                'author': self.author,
                'date': self.date,
                'url': self.url,
                'format_': self.format,
                'source_dir': self.source_dir,
                'meta_data': self.meta_data,
                'key_val': self.key_val,
                # 'fov': self.fov
                }
