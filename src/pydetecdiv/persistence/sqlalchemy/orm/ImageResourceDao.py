#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ImageResourceData data
"""
from sqlalchemy import Column, Integer, Float, Boolean, String, text, ForeignKey
from sqlalchemy.types import JSON
from sqlalchemy.orm import relationship, joinedload, composite
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base
import pydetecdiv.utils.ImageResource as ImageResource


class ImageResourceDao(DAO, Base):
    """
    DAO class for access to ImageResource records from the SQL database
    """
    __tablename__ = 'ImageResource'
    exclude = ['id_',]
    translate = {'shape': ('tdim', 'cdim', 'zdim', 'ydim', 'xdim'), }

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    uuid = Column(String(36))
    xdim = Column(Integer, nullable=False, server_default=text('512'))
    ydim = Column(Integer, nullable=False, server_default=text('512'))
    xyscale = Column(Float, nullable=False, server_default=text('1'))
    xyunit = Column(Float, nullable=False, server_default=text('1e-6'))  # reference unit in meters

    tdim = Column(Integer, nullable=False, server_default=text('1'))
    tscale = Column(Float, nullable=False, server_default=text('1'))
    tunit = Column(Float, nullable=False, server_default=text('60'))  # reference unit in seconds

    zdim = Column(Integer, nullable=False, server_default=text('1'))
    zscale = Column(Float, nullable=False, server_default=text('1'))
    zunit = Column(Float, nullable=False, server_default=text('1e-6'))  # reference unit in meters

    cdim = Column(Integer, nullable=False, server_default=text('1'))

    dims = composite(ImageResource.Dimension, tdim, cdim, zdim, ydim, xdim)

    fov = Column(Integer, ForeignKey('FOV.id_'), nullable=False, index=True)
    dataset = Column(Integer, ForeignKey('dataset.id_'), nullable=False, index=True)

    multi = Column(Boolean, nullable=False)

    key_val = Column(JSON)

    data_list_ = relationship('DataDao')

    def data_list(self, image_res_id: int) -> list[dict[str, object]]:
        """
        A method returning the list of Data records whose parent Dataset has id_ == dataset_id

        :param dataset_id: the id of the Dataset
        :type dataset_id: str
        :return: a list of Data records whose parent Dataset has id_ == dataset_id
        :rtype: list of dict
        """
        if self.session.query(ImageResourceDao).filter(ImageResourceDao.id_ == image_res_id).first() is not None:
            data_list = [data.record
                         for data in self.session.query(ImageResourceDao)
                         .options(joinedload(ImageResourceDao.data_list_))
                         .filter(ImageResourceDao.id_ == image_res_id)
                         .first().data_list_]
        else:
            data_list = []
        return data_list

    @property
    def record(self) -> dict[str, object]:
        """
        A method creating a DAO record dictionary from a fov row dictionary. This method is used to convert the SQL
        table columns into the FOV record fields expected by the domain layer

        :return: a FOV record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                # 'dims': self.dims,
                'xdim': self.xdim,
                'ydim': self.ydim,
                'zdim': self.zdim,
                'cdim': self.cdim,
                'tdim': self.tdim,
                'xyscale': self.xyscale,
                'xyunit': self.xyunit,
                'zscale': self.zscale,
                'zunit': self.zunit,
                'tscale': self.tscale,
                'tunit': self.tunit,
                'uuid': self.uuid,
                'fov': self.fov,
                'dataset': self.dataset,
                'multi': self.multi,
                'key_val': self.key_val,
                }
