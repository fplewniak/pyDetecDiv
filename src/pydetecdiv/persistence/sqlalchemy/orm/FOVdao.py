#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to FOV data
"""
from typing import Any

from sqlalchemy import Column, Integer, String
from sqlalchemy.types import JSON
from sqlalchemy.orm import relationship, joinedload
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class FOVdao(DAO, Base):
    """
    DAO class for access to FOV records from the SQL database
    """
    __tablename__ = 'FOV'
    exclude = ['id_', 'top_left', 'bottom_right']
    # translate = {'size': ('xsize', 'ysize'), }
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    uuid = Column(String(36))
    name = Column(String, unique=True, nullable=False)
    comments = Column(String)
    key_val = Column(JSON)

    roi_list_ = relationship('ROIdao')

    image_resources_ = relationship('ImageResourceDao')

    def image_resources(self, fov_id: int) -> list[dict[str, Any]]:
        """
        A method returning the list of ImageResource records whose parent FOV has id_ == fov_id

        :param fov_id: the id of the FOV
        :return: a list of ImageResource records whose parent FOV has id_ == fov_id
        """
        if self.session.query(FOVdao).filter(FOVdao.id_ == fov_id).first() is not None:
            imgres_list = [imgres.record
                         for imgres in self.session.query(FOVdao)
                         .options(joinedload(FOVdao.image_resources_))
                         .filter(FOVdao.id_ == fov_id)
                         .first().image_resources_]
        else:
            imgres_list = []
        return imgres_list

    @property
    def record(self) -> dict[str, Any]:
        """
        A method creating a DAO record dictionary from a fov row dictionary. This method is used to convert the SQL
        table columns into the FOV record fields expected by the domain layer

        :return: a FOV record as a dictionary with keys() appropriate for handling by the domain layer
        """
        return {'id_'     : self.id_,
                'name'    : self.name,
                'comments': self.comments,
                'uuid'    : self.uuid,
                'key_val' : self.key_val,
                }

    def roi_list(self, fov_id: int) -> list[dict[str, Any]]:
        """
        A method returning the list of ROI records whose parent FOV has id == fov_id

        :param fov_id: the id of the FOV
        :return: a list of ROI records with parent FOV id == fov_id
        """
        if self.session.query(FOVdao).filter(FOVdao.id_ == fov_id).first() is not None:
            roi_list = [roi.record
                        for roi in self.session.query(FOVdao)
                        .options(joinedload(FOVdao.roi_list_))
                        .filter(FOVdao.id_ == fov_id)
                        .first().roi_list_]
        else:
            roi_list = []
        return roi_list
