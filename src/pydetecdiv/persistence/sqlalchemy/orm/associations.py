#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
ORM classes describing associations between DAOs/tables
"""
from sqlalchemy import Column, ForeignKey
from sqlalchemy.orm import relationship
from pydetecdiv.persistence.sqlalchemy.orm.main import Base


class Linker:
    """
    A class providing methods for linking two Data access objects.
    """

    @staticmethod
    def association(class1_name, class2_name):
        """
        Defines the possible associations between DAO classes and returns the one corresponding to the classes passed
         as arguments.
        :return:
        """
        association_list = {
            ('FOVdao', 'DataDao'): FovData,
            ('DataDao', 'FOVdao'): FovData,
        }
        return association_list[(class1_name, class2_name)]

    @staticmethod
    def link(obj1, obj2):
        """
        Creates a link between the specified objects, calling the appropriate association object link method, as
        provided by the associations() method above.
        :param obj1: the first object to link
        :type obj1: DAO
        :param obj2: the second object to link
        :type obj2: DAO
        """
        Linker.association(obj1.__class__.__name__, obj2.__class__.__name__).link(obj1, obj2)


class FovData(Base):
    """
    Association many to many between FOV and Image data
    """
    __tablename__ = "FOVdata"
    fov = Column(ForeignKey("FOV.id_"), primary_key=True)
    data = Column(ForeignKey("data.id_"), primary_key=True)
    fov_ = relationship("FOVdao", back_populates='data_list', lazy='joined')
    data_ = relationship("DataDao", back_populates='fov_list_', lazy='joined')

    @staticmethod
    def fov_to_data():
        """
        Defines and returns the relationship from FOV to ImageData tables
        :return: the relationship between FOV and ImageData
        :rtype: sqlalchemy.orm.RelationshipProperty object
        """
        return relationship("FovData", back_populates='fov_', lazy='joined')

    @staticmethod
    def data_to_fov():
        """
        Defines and returns the relationship from ImageData to FOV tables
        :return: the relationship between ImageData and FOV
        :rtype: sqlalchemy.orm.RelationshipProperty object
        """
        return relationship("FovData", back_populates='data_', lazy='joined')

    @staticmethod
    def _link(obj1, obj2):
        """
        Creates a link between the specified FOVdao and DataDao objects.
        :param obj1: the first object to link
        :type obj1: FOVdao or DataDao
        :param obj2: the second object to link
        :type obj2: DataDao or FOVdao
        """
        a = FovData()
        if obj1.__class__.__name__ == 'FOVdao':
            a.data_ = obj2
            obj1.data_list.append(a)
        else:
            a.fov_ = obj2
            obj1.fov_list_.append(a)

    @staticmethod
    def link(obj1, obj2):
        """
        Checks there the existence of a link between the specified FOVdao and DataDao objects and creates a link
        (calling the _link method) only if there is no preexisting link.
        :param obj1: the first object to link
        :type obj1: FOVdao or DataDao
        :param obj2: the second object to link
        :type obj2: DataDao or FOVdao
        """
        if obj1.__class__.__name__ == 'FOVdao':
            query = obj1.session.query(FovData).filter(FovData.fov == obj1.id_).filter(FovData.data == obj2.id_)
        else:
            query = obj1.session.query(FovData).filter(FovData.fov == obj2.id_).filter(FovData.data == obj1.id_)
        if query.first() is None:
            FovData._link(obj1, obj2)

    @property
    def record(self):
        """
        A method creating a record dictionary from a FovData row dictionary.
        :return a FovData record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {
            'fov': self.fov,
            'data': self.data,
        }
