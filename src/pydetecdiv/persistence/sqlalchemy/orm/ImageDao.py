#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint, text
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base
import pydetecdiv.persistence.sqlalchemy.orm.dao as dao


class ImageDao(DAO, Base):
    """
    DAO class for access to Image records from the SQL database
    """
    __tablename__ = 'Image'
    __table_args__ = (
        UniqueConstraint('locator', 'c', 'z', 't', name='resource'),
    )
    exclude = ['id_', 'size', 'top_left', 'bottom_right']
    translate = {'drift': ('x_drift', 'y_drift'), }

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    image_data = Column(Integer, ForeignKey('ImageData.id_'), nullable=False, index=True)
    x_drift = Column(Integer, nullable=False, server_default=text('0'))
    y_drift = Column(Integer, nullable=False, server_default=text('0'))
    c = Column(Integer, nullable=False, server_default=text('0'))
    z = Column(Integer, nullable=False, server_default=text('0'))
    t = Column(Integer, nullable=False, server_default=text('0'))
    locator = Column(String, nullable=False, )
    mimetype = Column(String)

    def roi(self, image_id):
        """
        Return the ROI dao corresponding to the current image. The object is returned in a list for consistency.
        :param image_id: the id of the Image
        :return: the related ROI dao in a list
        :rtype: list of one ROIdao object
        """
        with self.session_maker() as session:
            roi = session.get(dao.ROIdao, (session.query(dao.ImageDataDao)
                                           .filter(dao.ImageDataDao.id_ == ImageDao.image_data)
                                           .filter(ImageDao.id_ == image_id).first().roi)).record
        return [roi]

    def fov(self, image_id):
        """
        Return the FOV dao corresponding to the current image. The object is returned in a list for consistency.
        :param image_id: the id of the Image
        :return: the related FOV dao in a list
        :rtype: list of one FOVdao object
        :param image_id:
        :return:
        """
        with self.session_maker() as session:
            fov = session.get(dao.FOVdao, (session.get(dao.ROIdao, (session.query(dao.ImageDataDao)
                                                                    .filter(dao.ImageDataDao.id_ == ImageDao.image_data)
                                                                    .filter(
                ImageDao.id_ == image_id).first().roi)).fov)).record
        return [fov]

    @property
    def record(self):
        """
        A method creating a record dictionary from an Image row dictionary. This method is used to convert the SQL
        table columns into the Image record fields expected by the domain layer
        :return an Image record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'image_data': self.image_data,
                'drift': (self.x_drift, self.y_drift),
                'c': self.c,
                'z': self.z,
                't': self.t,
                'locator': self.locator,
                'mimetype': self.mimetype,
                }
