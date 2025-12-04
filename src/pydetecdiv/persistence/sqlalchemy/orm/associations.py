#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
ORM classes describing associations between DAOs/tables
"""
from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from pydetecdiv.persistence.sqlalchemy.orm.DataDao import DataDao
    from pydetecdiv.persistence.sqlalchemy.orm.ROIdao import ROIdao

from sqlalchemy import Column, ForeignKey
from sqlalchemy.orm import relationship, Relationship

from pydetecdiv.persistence.sqlalchemy.orm.main import Base, DAO

DAOvar = TypeVar('DAOvar', bound=DAO)


class ROIdata(Base):
    """
    Association many to many between ROI and Image data
    """
    __tablename__ = "ROIdata"
    roi = Column(ForeignKey("ROI.id_"), primary_key=True)
    data = Column(ForeignKey("data.id_"), primary_key=True)
    roi_ = relationship("ROIdao", back_populates='data_list', lazy='joined')
    data_ = relationship("DataDao", back_populates='roi_list_', lazy='joined')

    @staticmethod
    def roi_to_data() -> Relationship:
        """
        Defines and returns the relationship from ROI to ImageData tables

        :return: the relationship between ROI and ImageData
        """
        return relationship("ROIdata", back_populates='roi_', lazy='joined')

    @staticmethod
    def data_to_roi() -> Relationship:
        """
        Defines and returns the relationship from ImageData to ROI tables

        :return: the relationship between ImageData and ROI
        """
        return relationship("ROIdata", back_populates='data_', lazy='joined')

    @staticmethod
    def _link(obj1: 'ROIdao | DataDao', obj2: 'ROIdao | DataDao') -> None:
        """
        Creates a link between the specified ROIdao and DataDao objects.

        :param obj1: the first object to link
        :param obj2: the second object to link
        """
        a = ROIdata()
        if obj1.__class__.__name__ == 'ROIdao':
            a.data_ = obj2
            obj1.data_list.append(a)
        else:
            a.roi_ = obj2
            obj1.roi_list_.append(a)

    @staticmethod
    def link(obj1: 'ROIdao | DataDao', obj2: 'ROIdao | DataDao') -> None:
        """
        Checks there the existence of a link between the specified ROIdao and DataDao objects and creates a link
        (calling the _link method) only if there is no preexisting link.

        :param obj1: the first object to link
        :param obj2: the second object to link
        """
        if obj1.__class__.__name__ == 'ROIdao':
            query = obj1.session.query(ROIdata).filter(ROIdata.roi == obj1.id_).filter(ROIdata.data == obj2.id_)
        else:
            query = obj1.session.query(ROIdata).filter(ROIdata.roi == obj2.id_).filter(ROIdata.data == obj1.id_)
        if query.first() is None:
            ROIdata._link(obj1, obj2)

    @property
    def record(self) -> dict[str, Column]:
        """
        A method creating a record dictionary from a ROIdata row dictionary.

        :return: a ROIdata record as a dictionary with keys() appropriate for handling by the domain layer
        """
        return {
            'roi' : self.roi,
            'data': self.data,
            }


class Linker:
    """
    A class providing methods for linking two Data access objects.
    """

    @staticmethod
    def association(class1_name: str, class2_name: str) -> type[ROIdata]:
        """
        Defines the possible associations between DAO classes and returns the one corresponding to the classes passed
        as arguments.
        """
        association_list = {
            # ('FOVdao', 'DataDao'): FovData,
            # ('DataDao', 'FOVdao'): FovData,
            ('ROIdao', 'DataDao'): ROIdata,
            ('DataDao', 'ROIdao'): ROIdata,
            }
        return association_list[(class1_name, class2_name)]

    @staticmethod
    def link(obj1: DAOvar, obj2: DAOvar):
        """
        Creates a link between the specified objects, calling the appropriate association object link method, as
        provided by the associations() method above.

        :param obj1: the first object to link
        :param obj2: the second object to link
        """
        Linker.association(obj1.__class__.__name__, obj2.__class__.__name__).link(obj1, obj2)
