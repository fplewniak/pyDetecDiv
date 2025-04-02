from typing import Any

from sqlalchemy import Column, Integer, String, JSON, ForeignKey

from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class EntityDao(DAO, Base):
    """
    DAO class for access to Entity records from the SQL database
    """
    __tablename__ = 'Entity'
    exclude = ['id_']
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    uuid = Column(String(36))
    roi = Column(Integer, ForeignKey('ROI.id_'), nullable=False, index=True)
    name = Column(String, unique=True, nullable=False)
    category = Column(String)
    key_val = Column(JSON)

    @property
    def record(self) -> dict[str, Any]:
        """
        A method creating a record dictionary from a roi row dictionary. This method is used to convert the SQL
        table columns into the ROI record fields expected by the domain layer

        :return: a ROI record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_'     : self.id_,
                'uuid'    : self.uuid,
                'roi'     : self.roi,
                'name'    : self.name,
                'category': self.category,
                'key_val' : self.key_val,
                }
