#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to Entity data
"""
from typing import Any

from sqlalchemy import Column, Integer, String, JSON, ForeignKey, text

from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class BoundingBoxDao(DAO, Base):
    """
    DAO class for access to Entity records from the SQL database
    """
    __tablename__ = 'BoundingBox'
    exclude = ['id_']
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    uuid = Column(String(36))
    entity = Column(Integer, ForeignKey('Entity.id_'), nullable=False, index=True)
    name = Column(String, unique=False, nullable=False)
    frame = Column(Integer, nullable=False, server_default=text('0'))
    x = Column(Integer, nullable=False, server_default=text('0'))
    y = Column(Integer, nullable=False, server_default=text('0'))
    width = Column(Integer, nullable=False, server_default=text('-1'))
    height = Column(Integer, nullable=False, server_default=text('-1'))
    key_val = Column(JSON)

    @property
    def record(self) -> dict[str, Any]:
        """
        A method creating a record dictionary from an entity row dictionary. This method is used to convert the SQL
        table columns into the Entity record fields expected by the domain layer

        :return: an Entity record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_'    : self.id_,
                'uuid'   : self.uuid,
                'entity' : self.entity,
                'name'   : self.name,
                'frame'  : self.frame,
                'x'      : self.x,
                'y'      : self.y,
                'width'  : self.width,
                'height' : self.height,
                'key_val': self.key_val,
                }
