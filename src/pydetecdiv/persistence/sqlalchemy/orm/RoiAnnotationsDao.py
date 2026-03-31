from typing import Any

from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import relationship, joinedload

from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class RoiAnnotationsDao(DAO, Base):
    """
    DAO class for access to ROI records from the SQL database
    """
    __tablename__ = 'RoiAnnotations'
    exclude = ['id_', ]
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    roi = Column(Integer, ForeignKey('ROI.id_'), nullable=False, index=True)
    t = Column(Integer, nullable=False, index=True)
    annotation = Column(Integer, nullable=False, index=True)
    classification = Column(Integer, ForeignKey('Classification.id_'), nullable=False, index=True)
    run = Column(Integer, ForeignKey('run.id_'), nullable=False, index=True)
    uuid = Column(String(36))
    key_val = Column(JSON)

    @property
    def record(self) -> dict[str, Any]:
        """
        A method creating a record dictionary. This method is used to convert the SQL
        table columns into the record fields expected by the domain layer

        :return: a record as a dictionary with keys() appropriate for handling by the domain layer
        """
        return {'id_'           : self.id_,
                'roi'           : self.roi,
                't'             : self.t,
                'annotation'    : self.annotation,
                'classification': self.classification,
                'run'           : self.run,
                'uuid'          : self.uuid,
                'key_val'       : self.key_val,
                }
