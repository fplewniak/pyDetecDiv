#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to FOV data
"""
import itertools

from sqlalchemy import Column, Integer, String, text, select
from sqlalchemy.orm import relationship, joinedload
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base
import pydetecdiv.persistence.sqlalchemy.orm.dao as dao


class FOVdao(DAO, Base):
    """
    DAO class for access to FOV records from the SQL database
    """
    __tablename__ = 'FOV'
    exclude = ['id_', 'top_left', 'bottom_right']
    translate = {'size': ('xsize', 'ysize'), }

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    name = Column(String, unique=True, nullable=False)
    comments = Column(String)
    xsize = Column(Integer, nullable=False, server_default=text('1000'))
    ysize = Column(Integer, nullable=False, server_default=text('1000'))

    roi_list_ = relationship('ROIdao')

    # image_data_list = FovData.fov_to_image_data()

    @property
    def record(self):
        """
        A method creating a DAO record dictionary from a fov row dictionary. This method is used to convert the SQL
        table columns into the FOV record fields expected by the domain layer
        :return a FOV record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'name': self.name,
                'comments': self.comments,
                'top_left': (0, 0),
                'bottom_right': (self.xsize - 1, self.ysize - 1),
                'size': (self.xsize, self.ysize),
                }

    def image_data(self, fov_id):
        """
        A method returning the list of Image data object records linked to FOV with id_ == fov_id
        :param fov_id: the id of the FOV
        :type fov_id: int
        :return: a list of ImageData records linked to FOV with id_ == fov_id
        :rtype: list
        """
        with self.session_maker() as session:
            image_data = [i.record for i in itertools.chain(
                *[roi.image_data_list for roi in session.query(dao.ROIdao).filter(dao.ROIdao.fov == fov_id).all()])]
        return image_data

    def roi_list(self, fov_id):
        """
        A method returning the list of ROI records whose parent FOV has id == fov_id
        :param fov_id: the id of the FOV
        :type fov_id: int
        :return: a list of ROI records with parent FOV id == fov_id
        :rtype: list
        """
        with self.session_maker() as session:
            roi_list = [roi.record
                        for roi in session.query(FOVdao)
                        .options(joinedload(FOVdao.roi_list_))
                        .filter(FOVdao.id_ == fov_id)
                        .first().roi_list_]
        return roi_list

    def image_list(self, fov_id):
        """
        A method returning the Image records linked to FOV with id_ == fov_id
        :param fov_id: the id of the Image data
        :type fov_id: int
        :return: a list containing the Image records linked to FOV with id_ == fov_id
        :rtype: list
        """
        with self.session_maker() as session:
            image_list = [image.record for image in
                          session.scalars(
                              select(dao.ImageDao)
                              .where(dao.ImageDataDao.id_ == dao.ImageDao.image_data)
                              .where(dao.ROIdao.id_ == dao.ImageDataDao.roi)
                              .where(fov_id == dao.ROIdao.fov)
                          )]
        return image_list
