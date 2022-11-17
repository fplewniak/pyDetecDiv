#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from sqlalchemy import Column, Integer, String, ForeignKey, text
from sqlalchemy.orm import joinedload, relationship
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base
import pydetecdiv.persistence.sqlalchemy.orm.dao as dao


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

    image_data_list = relationship('ImageDataDao')

    @property
    def record(self):
        """
        A method creating a record dictionary from a roi row dictionary. This method is used to convert the SQL
        table columns into the ROI record fields expected by the domain layer
        :return a ROI record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'name': self.name,
                'fov': self.fov,
                'top_left': (self.x0_, self.y0_),
                'bottom_right': (self.x1_, self.y1_),
                'size': (self.x1_ - self.x0_ + 1, self.y1_ - self.y0_ + 1)
                }

    def image_data(self, roi_id):
        """
        A method returning the list of Image data object records linked to ROI with id_ == roi_id
        :param roi_id: the id of the ROI
        :type roi_id: int
        :return: a list of ImageData records linked to ROI with id_ == roi_id
        :rtype: list
        """
        image_data = [image_data.record
                      for image_data in self.session.query(ROIdao)
                      .options(joinedload(ROIdao.image_data_list))
                      .filter(ROIdao.id_ == roi_id)
                      .first().image_data_list]
        return image_data

    def image_list(self, roi_id):
        """
        A method returning the Image records linked to ImageData with id_ == roi_id
        :param roi_id: the id of the ROI
        :type roi_id: int
        :return: a list containing the Image records linked to ROI with id_ == roi_id
        :rtype: list
        """
        image_list = [image.record for image in
                      self.session.query(dao.ImageDao)
                      .filter(dao.ImageDataDao.id_ == dao.ImageDao.image_data)
                      .filter(roi_id == dao.ImageDataDao.roi)
                      .all()]
        return image_list
