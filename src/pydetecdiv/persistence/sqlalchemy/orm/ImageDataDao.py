#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Image data access DAO
"""
from sqlalchemy import text, Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import joinedload, relationship
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base
from pydetecdiv.persistence.sqlalchemy.orm.associations import FovData, RoiData


class ImageDataDao(DAO, Base):
    """
    DAO class for access to ImageData records from the SQL database
    """
    __tablename__ = 'ImageData'
    exclude = ['id_', 'stacks', 'videos']
    translate = {'top_left': ('x0_', 'y0_'), 'bottom_right': ('x1_', 'y1_'), 'file_resource': 'resource'}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    name = Column(String, unique=True, )
    channel = Column(Integer, nullable=False, )
    x0_ = Column(Integer, nullable=False, server_default=text('0'))
    y0_ = Column(Integer, nullable=False, server_default=text('-1'))
    x1_ = Column(Integer, nullable=False, server_default=text('0'))
    y1_ = Column(Integer, nullable=False, server_default=text('-1'))
    # stacks = Column(Integer, nullable=False, server_default=text('1'))
    # frames = Column(Integer, nullable=False, server_default=text('1'))
    stack_interval = Column(Float, )
    frame_interval = Column(Float, )
    orderdims = Column(String, nullable=False, server_default=text('xyzct'))
    resource = Column(Integer, ForeignKey('FileResource.id_'), nullable=False, index=True)
    path = Column(String, )
    mimetype = Column(String)

    image_list_ = relationship('ImageDao')

    fov_list_ = FovData.image_data_to_fov()
    roi_list_ = RoiData.image_data_to_roi()

    def fov_list(self, image_data_id):
        """
        A method returning the list of FOV object records linked to ImageData with id_ == image_data_id
        :param image_data_id: the id of the Image data
        :type image_data_id: int
        :return: a list of FOV records linked to ImageData with id_ == image_data_id
        :rtype: list
        """
        with self.session_maker() as session:
            fov_list = [association.fov_.record
                        for association in session.query(ImageDataDao)
                        .options(joinedload(ImageDataDao.fov_list_))
                        .filter(ImageDataDao.id_ == image_data_id)
                        .first().fov_list_]
        return fov_list

    def roi_list(self, image_data_id):
        """
        A method returning the list of ROI object records linked to ImageData with id_ == image_data_id
        :param image_data_id: the id of the Image data
        :type image_data_id: int
        :return: a list of ROI records linked to ImageData with id_ == image_data_id
        :rtype: list
        """
        with self.session_maker() as session:
            roi_list = [association.roi_.record
                        for association in session.query(ImageDataDao)
                        .options(joinedload(ImageDataDao.roi_list_))
                        .filter(ImageDataDao.id_ == image_data_id)
                        .first().roi_list_]
        return roi_list

    def image_list(self, image_data_id):
        """
        A method returning the list of ROI records whose parent FOV has id == fov_id
        :param fov_id: the id of the FOV
        :type fov_id: int
        :return: a list of ROI records with parent FOV id == fov_id
        :rtype: list
        """
        with self.session_maker() as session:
            image_list = [image.record
                          for image in session.query(ImageDataDao)
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
                'top_left': (self.x0_, self.y0_),
                'bottom_right': (self.x1_, self.y1_),
                'channel': self.channel,
                'stack_interval': self.stack_interval,
                'frame_interval': self.frame_interval,
                'orderdims': self.orderdims,
                'file_resource': self.resource,
                'path': self.path,
                'mimetype': self.mimetype,
                }
