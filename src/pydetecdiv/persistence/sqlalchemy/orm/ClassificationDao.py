from typing import Any

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
    runs_ = relationship('RunDao', secondary=RoiAnnotationsDao.__table__, viewonly=True)

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

    def roi_annotations(self, classifier_id: int) -> list[dict[str, object]]:
        """
        A method returning the list of ROI Annotations records whose parent Classifier has id_ == classifier_id

        :param classifier_id: the id of the Classifier
        :return: a list of ROI Annotations records with parent Classifier id_ == classifier_id
        """
        if self.session.query(ClassificationDao).filter(ClassificationDao.id_ == classifier_id).first() is not None:
            annotations = [annotation.record
                           for annotation in self.session.query(ClassificationDao)
                           .options(joinedload(ClassificationDao.roi_annotations_))
                           .filter(ClassificationDao.id_ == classifier_id)
                           .first().roi_annotations_]
        else:
            annotations = []
        return annotations

    def runs(self, classifier_id: int) -> list[dict[str, object]]:
        """
        A method returning the list of Run records whose parent Classifier has id_ == classifier_id

        :param classifier_id: the id of the Classifier
        :return: a list of Run records with parent Classifier id_ == classifier_id
        """
        if self.session.query(ClassificationDao).filter(ClassificationDao.id_ == classifier_id).first() is not None:
            runs = [run.record
                    for run in self.session.query(RunDao)
                    .options(joinedload(ClassificationDao.runs_))
                    .filter(ClassificationDao.id_ == classifier_id)
                    .first().runs_]
        else:
            runs = []
        return runs
