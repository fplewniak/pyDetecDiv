#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from sqlalchemy import Column, Integer, ForeignKey, text
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class ImageDao(DAO, Base):
    """
    DAO class for access to Image records from the SQL database
    """
    __tablename__ = 'Image'
    exclude = ['id_', 'size', 'top_left', 'bottom_right']
    translate = {'drift': ('x_drift', 'y_drift'),}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    image_data = Column(Integer, ForeignKey('ImageData.id_'), nullable=False, index=True)
    x_drift = Column(Integer, nullable=False, server_default=text('0'))
    y_drift = Column(Integer, nullable=False, server_default=text('0'))
    z = Column(Integer, nullable=False, server_default=text('0'))
    t = Column(Integer, nullable=False, server_default=text('0'))
    resource = Column(Integer, ForeignKey('FileResource.id_'), nullable=False, index=True)

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
                'z': self.z,
                't': self.t,
                'resource': self.resource,
                }
