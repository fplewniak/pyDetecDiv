#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to Entity data
"""
from typing import Any

from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import joinedload, relationship

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
    exit_frame = Column(Integer, nullable=False)
    key_val = Column(JSON)

    bounding_boxes_ = relationship('BoundingBoxDao')
    masks_ = relationship('MaskDao')
    points_ = relationship('PointDao')

    def bounding_boxes(self, entity_id: int) -> list[dict[str, object]]:
        """
        A method returning the list of BoundingBox records whose parent Entity has id == entity_id

        :param entity_id: the id of the Entity
        :type entity_id: int
        :return: a list of BoundingBox records with parent Entity_id id == entity_id
        :rtype: list
        """
        if self.session.query(EntityDao).filter(EntityDao.id_ == entity_id).first() is not None:
            bounding_boxes = [bounding_box.record
                              for bounding_box in self.session.query(EntityDao)
                              .options(joinedload(EntityDao.bounding_boxes_))
                              .filter(EntityDao.id_ == entity_id)
                              .first().bounding_boxes_]
        else:
            bounding_boxes = []
        return bounding_boxes

    def masks(self, entity_id: int) -> list[dict[str, object]]:
        """
        A method returning the list of Mask records whose parent Entity has id == entity_id

        :param entity_id: the id of the Entity
        :type entity_id: int
        :return: a list of Mask records with parent Entity_id id == entity_id
        :rtype: list
        """
        if self.session.query(EntityDao).filter(EntityDao.id_ == entity_id).first() is not None:
            masks = [mask.record
                     for mask in self.session.query(EntityDao)
                     .options(joinedload(EntityDao.masks_))
                     .filter(EntityDao.id_ == entity_id)
                     .first().masks_]
        else:
            masks = []
        return masks

    def points(self, entity_id: int) -> list[dict[str, object]]:
        """
        A method returning the list of Mask records whose parent Entity has id == entity_id

        :param entity_id: the id of the Entity
        :type entity_id: int
        :return: a list of Mask records with parent Entity_id id == entity_id
        :rtype: list
        """
        if self.session.query(EntityDao).filter(EntityDao.id_ == entity_id).first() is not None:
            points = [point.record
                      for point in self.session.query(EntityDao)
                      .options(joinedload(EntityDao.points_))
                      .filter(EntityDao.id_ == entity_id)
                      .first().points_]
        else:
            points = []
        return points

    @property
    def record(self) -> dict[str, Any]:
        """
        A method creating a record dictionary from an entity row dictionary. This method is used to convert the SQL
        table columns into the Entity record fields expected by the domain layer

        :return: an Entity record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_'       : self.id_,
                'uuid'      : self.uuid,
                'roi'       : self.roi,
                'name'      : self.name,
                'category'  : self.category,
                'exit_frame': self.exit_frame,
                'key_val'   : self.key_val,
                }
