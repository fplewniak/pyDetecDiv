"""
Access to Classification schemes data
"""
from typing import Any

import sqlalchemy
from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy.orm import relationship, joinedload

from pydetecdiv.persistence.sqlalchemy.orm.RoiAnnotationsDao import RoiAnnotationsDao
from pydetecdiv.persistence.sqlalchemy.orm.RunDao import RunDao
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class ClassificationDao(DAO, Base):
    """
    DAO class for access to ROI records from the SQL database
    """
    __tablename__ = 'Classification'
    exclude = ['id_', ]
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    name = Column(String, unique=True, nullable=False)
    classes = Column(JSON, nullable=False)
    uuid = Column(String(36))
    key_val = Column(JSON)

    roi_annotations_ = relationship('RoiAnnotationsDao', viewonly=True)

    @property
    def record(self) -> dict[str, Any]:
        """
        A method creating a record dictionary. This method is used to convert the SQL
        table columns into the record fields expected by the domain layer

        :return: a record as a dictionary with keys() appropriate for handling by the domain layer
        """
        return {'id_'    : self.id_,
                'name'   : self.name,
                'classes': self.classes,
                'uuid'   : self.uuid,
                'key_val': self.key_val,
                }

    def annotations(self, classification_id: int) -> list[dict[str, Any]]:
        """
        A method returning the list of ROI Annotations records whose parent Classification has id_ == classification_id

        :param classification_id: the id of the Classification
        :return: a list of ROI Annotations records with parent Classification id_ == classification_id
        """
        if self.session.query(ClassificationDao).filter(ClassificationDao.id_ == classification_id).first() is not None:
            annotations = [annotation.record
                           for annotation in self.session.query(ClassificationDao)
                           .options(joinedload(ClassificationDao.roi_annotations_))
                           .filter(ClassificationDao.id_ == classification_id)
                           .first().roi_annotations_]
        else:
            annotations = []
        return annotations

    def runs(self, classification_id: int) -> list[dict[str, Any]]:
        """
        A method returning the list of Run records whose parent Classification has id_ == classification_id

        :param classification_id: the id of the Classification
        :return: a list of Run records with parent Classification id_ == classification_id
        """
        if self.session.query(ClassificationDao).filter(ClassificationDao.id_ == classification_id).first() is not None:
            stmt = (sqlalchemy.select(RunDao).join_from(RoiAnnotationsDao, ClassificationDao)
                    .where(ClassificationDao.id_ == classification_id)
                    .where(ClassificationDao.id_ == RoiAnnotationsDao.classification)
                    .where(RunDao.id_ == RoiAnnotationsDao.run))

            runs = [run.record for run in self.session.execute(stmt).unique().scalars()]
        else:
            runs = []
        return runs
