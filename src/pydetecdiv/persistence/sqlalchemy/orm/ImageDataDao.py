#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Image data access DAO
"""
from sqlalchemy import text, Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import joinedload, relationship
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base
import pydetecdiv.persistence.sqlalchemy.orm.dao as dao


class ImageDataDao(DAO, Base):
    """
    DAO class for access to ImageData records from the SQL database
    """
    __tablename__ = 'ImageData'
    exclude = ['id_', 'stacks', 'videos']
    translate = {'shape': {}}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    roi = Column(Integer, ForeignKey('ROI.id_'), nullable=False, index=True)
    name = Column(String, )
    x = Column(Integer, nullable=False, server_default=text('1000'))
    y = Column(Integer, nullable=False, server_default=text('1000'))
    z = Column(Integer, nullable=False, server_default=text('1'))
    c = Column(Integer, nullable=False, server_default=text('1'))
    t = Column(Integer, nullable=False, server_default=text('1'))
    stack_interval = Column(Float, )
    frame_interval = Column(Float, )
    orderdims = Column(String, nullable=False, server_default=text('xyzct'))

    image_list_ = relationship('ImageDao')

    def fov(self, image_data_id):
        """
        A method returning the FOV object record linked to ImageData with id_ == image_data_id
        :param image_data_id: the id of the Image data
        :type image_data_id: int
        :return: a list containing the FOV record linked to ImageData with id_ == image_data_id
        :rtype: list
        """
        fov = self.session.query(dao.FOVdao).filter(ImageDataDao.id_ == image_data_id).filter(
            dao.FOVdao.id_ == dao.ROIdao.fov).filter(dao.ROIdao.id_ == ImageDataDao.roi).first().record
        return [fov]

    def image_list(self, image_data_id):
        """
        A method returning the list of ROI records whose parent FOV has id == fov_id
        :param fov_id: the id of the FOV
        :type fov_id: int
        :return: a list of ROI records with parent FOV id == fov_id
        :rtype: list
        """
        image_list = [image.record
                      for image in self.session.query(ImageDataDao)
                      .options(joinedload(ImageDataDao.image_list_))
                      .filter(ImageDataDao.id_ == image_data_id)
                      .first().image_list_]
        return image_list

    @property
    def record(self):
        """
        A method creating a record dictionary from a image data row dictionary. This method is used to convert the SQL
        table columns into the image data record fields expected by the domain layer
        :return an image data record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'name': self.name,
                'roi': self.roi,
                'shape': tuple(self.__getattribute__(v) for v in self.orderdims),
                'stack_interval': self.stack_interval,
                'frame_interval': self.frame_interval,
                'orderdims': self.orderdims,
                }
