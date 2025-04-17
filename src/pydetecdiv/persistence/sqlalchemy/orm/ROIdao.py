#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from typing import Any

from sqlalchemy import Column, Integer, String, ForeignKey, text
from sqlalchemy.types import JSON
from sqlalchemy.orm import joinedload, relationship
from pydetecdiv.persistence.sqlalchemy.orm.associations import ROIdata
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

    # image_data_list = relationship('ImageDataDao')
    data_list = ROIdata.roi_to_data()

    entities_ = relationship('EntityDao')

    @property
    def record(self) -> dict[str, Any]:
        """
        A method creating a record dictionary from a roi row dictionary. This method is used to convert the SQL
        table columns into the ROI record fields expected by the domain layer

        :return: a ROI record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'name': self.name,
                'fov': self.fov,
                'top_left': (self.x0_, self.y0_),
                'bottom_right': (self.x1_, self.y1_),
                'size': (self.x1_ - self.x0_ + 1, self.y1_ - self.y0_ + 1),
                'uuid': self.uuid,
                'key_val': self.key_val,
                }

    def data(self, roi_id: int) -> list[dict[str, Any]]:
        """
        Returns a list of DataDao objects linked to the ROIdao object with the specified id_

        :param roi_id: the id_ of the ROI
        :type roi_id: int
        :return: the list of Data records linked to the ROI
        :type: list of dict
        """
        return [i.record
                for i in self.session.query(dao.DataDao)
                .filter(ROIdata.data == dao.DataDao.id_)
                .filter(ROIdata.roi == roi_id)
                ]

    def entities(self, roi_id: int) -> list[dict[str, object]]:
        """
        A method returning the list of Entity records whose parent ROI has id == roi_id

        :param roi_id: the id of the ROI
        :type roi_id: int
        :return: a list of Entity records with parent ROI id == roi_id
        :rtype: list
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

    # def image_data(self, roi_id):
    #     """
    #     A method returning the list of Image data object records linked to ROI with id_ == roi_id
    #     :param roi_id: the id of the ROI
    #     :type roi_id: int
    #     :return: a list of ImageData records linked to ROI with id_ == roi_id
    #     :rtype: list
    #     """
    #     image_data = [image_data.record
    #                   for image_data in self.session.query(ROIdao)
    #                   .options(joinedload(ROIdao.image_data_list))
    #                   .filter(ROIdao.id_ == roi_id)
    #                   .first().image_data_list]
    #     return image_data
    #
    # def image_list(self, roi_id):
    #     """
    #     A method returning the Image records linked to ImageData with id_ == roi_id
    #     :param roi_id: the id of the ROI
    #     :type roi_id: int
    #     :return: a list containing the Image records linked to ROI with id_ == roi_id
    #     :rtype: list
    #     """
    #     image_list = [image.record for image in
    #                   self.session.query(dao.ImageDao)
    #                   .filter(dao.ImageDataDao.id_ == dao.ImageDao.image_data)
    #                   .filter(roi_id == dao.ImageDataDao.roi)
    #                   .all()]
    #     return image_list
