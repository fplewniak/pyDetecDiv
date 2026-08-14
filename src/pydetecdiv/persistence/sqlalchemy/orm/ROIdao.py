#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from typing import Any

import sqlalchemy
from sqlalchemy import Column, Integer, String, ForeignKey, text
from sqlalchemy.types import JSON
from sqlalchemy.orm import joinedload, relationship

from pydetecdiv.persistence.sqlalchemy.orm.RoiAnnotationsDao import RoiAnnotationsDao
# from pydetecdiv.persistence.sqlalchemy.orm.associations import ROIdata
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base
from pydetecdiv.persistence.sqlalchemy.orm import dao


class ROIdao(DAO, Base):
    """
    DAO class for access to ROI records from the SQL database
    """
    __tablename__ = 'ROI'
    exclude = ['id_', 'size', ]
    translate = {'top_left': ('x0_', 'y0_'), 'bottom_right': ('x1_', 'y1_')}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    name = Column(String, unique=True, nullable=False)
    fov = Column(Integer, ForeignKey('FOV.id_'), nullable=False, index=True)
    x0_ = Column(Integer, nullable=False, server_default=text('0'))
    y0_ = Column(Integer, nullable=False, server_default=text('0'))
    x1_ = Column(Integer, nullable=False, server_default=text('-1'))
    y1_ = Column(Integer, nullable=False, server_default=text('-1'))
    uuid = Column(String(36))
    key_val = Column(JSON)

    # data_list = ROIdata.roi_to_data()

    entities_ = relationship('EntityDao')

    @property
    def record(self) -> dict[str, Any]:
        """
        A method creating a record dictionary from a roi row dictionary. This method is used to convert the SQL
        table columns into the ROI record fields expected by the domain layer

        :return: a ROI record as a dictionary with keys() appropriate for handling by the domain layer
        """
        return {'id_'         : self.id_,
                'name'        : self.name,
                'fov'         : self.fov,
                'top_left'    : (self.x0_, self.y0_),
                'bottom_right': (self.x1_, self.y1_),
                'size'        : (self.x1_ - self.x0_ + 1, self.y1_ - self.y0_ + 1),
                'uuid'        : self.uuid,
                'key_val'     : self.key_val,
                }

    # def data(self, roi_id: int) -> list[dict[str, Any] | property]:
    #     """
    #     Returns a list of DataDao objects linked to the ROIdao object with the specified id_
    #
    #     :param roi_id: the id_ of the ROI
    #     :return: the list of Data records linked to the ROI
    #     """
    #     return [i.record
    #             for i in self.session.query(dao.DataDao)
    #             .filter(ROIdata.data == dao.DataDao.id_)
    #             .filter(ROIdata.roi == roi_id)
    #             ]

    def entities(self, roi_id: int) -> list[dict[str, Any]]:
        """
        A method returning the list of Entity records whose parent ROI has id_ == roi_id

        :param roi_id: the id of the ROI
        :return: a list of Entity records with parent ROI id_ == roi_id
        """
        if self.session.query(ROIdao).filter(ROIdao.id_ == roi_id).first() is not None:
            entities = [entity.record
                        for entity in self.session.query(ROIdao)
                        .options(joinedload(ROIdao.entities_))
                        .filter(ROIdao.id_ == roi_id)
                        .first().entities_]
        else:
            entities = []
        return entities

    def annotations(self, roi_id: int) -> list[dict[str, Any]]:
        """
        A method returning the list of Entity records whose parent ROI has id_ == roi_id

        :param roi_id: the id of the ROI
        :return: a list of Entity records with parent ROI id_ == roi_id
        """
        if self.session.query(ROIdao).filter(ROIdao.id_ == roi_id).first() is not None:
            stmt = (sqlalchemy.select(RoiAnnotationsDao).join(ROIdao)
                    .where(roi_id == RoiAnnotationsDao.roi).where(ROIdao.id_ == roi_id))

            annotations = [annotation.record for annotation in self.session.execute(stmt).unique().scalars()]
        else:
            annotations = []
        return annotations
