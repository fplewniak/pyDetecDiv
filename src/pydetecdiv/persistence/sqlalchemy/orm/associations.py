#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
ORM classes describing associations between DAOs/tables
"""
from sqlalchemy import Column, ForeignKey
from sqlalchemy.orm import relationship
from pydetecdiv.persistence.sqlalchemy.orm.main import Base


class Linker:
    @staticmethod
    def associations():
        return {
            ('FOVdao', 'ImageDataDao'): FovData,
            ('ImageDataDao', 'FOVdao'): FovData,
        }

    @staticmethod
    def link(obj1, obj2):
        return Linker.associations()[(obj1.__class__.__name__, obj2.__class__.__name__)].link(obj1, obj2)

    @staticmethod
    def unlink(obj1, obj2):
        ...


class FovData(Base):
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

    @staticmethod
    def link(obj1, obj2):
        a = FovData()
        if obj1.__class__.__name__ == 'FOVdao':
            a.image_data_ = obj2
            obj1.image_data_list.append(a)
        else:
            a.fov_ = obj2
            obj1.fov_list_.append(a)

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
