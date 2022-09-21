#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
ORM classes describing associations between DAOs/tables
"""
from sqlalchemy import Column, ForeignKey
from sqlalchemy.orm import relationship
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class FovData(DAO, Base):
    """
    Association many to many between FOV and Image data
    """
    __tablename__ = "FOVdata"
    fov = Column(ForeignKey("FOV.id_"), primary_key=True)
    image_data = Column(ForeignKey("ImageData.id_"), primary_key=True)
    fov_ = relationship("FOVdao", back_populates='image_data_list', lazy='joined')
    image_data_ = relationship("ImageDataDao", back_populates='fov_list_', lazy='joined')

    @staticmethod
    def fov_to_image_data():
        return relationship("FovData", back_populates='fov_', lazy='joined')

    @staticmethod
    def image_data_to_fov():
        return relationship("FovData", back_populates='image_data_', lazy='joined')

    @property
    def record(self):
        """
        A method creating a record dictionary from a FovData row dictionary.
        :return a FovData record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {
            'fov': self.fov,
            'image_data': self.image_data,
        }
