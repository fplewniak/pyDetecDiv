#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to File resource data
"""
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship, joinedload
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class FileResourceDao(DAO, Base):
    """
    DAO class for access to FileReource records from the SQL database
    """
    __tablename__ = 'FileResource'
    exclude = ['id_', ]
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    locator = Column(String, unique=True, )
    mimetype = Column(String)

    image_data_ = relationship('ImageDataDao')

    def image_data(self, resource_id):
        """
        A method returning the list of Image data objects in File resource with id_ == resource_id
        :param resource_id: the id of the file resource
        :type resource_id: int
        :return: a list of ImageData records with parent File resource id_ == resource_id
        :rtype: list
        """
        with self.session_maker() as session:
            image_data = [image_data.record
                          for image_data in session.query(FileResourceDao)
                          .options(joinedload(FileResourceDao.image_data_))
                          .filter(FileResourceDao.id_ == resource_id)
                          .first().image_data_]
        return image_data

    @property
    def record(self):
        """
        A method creating a record dictionary from a file resource row dictionary. This method is used to convert the
        SQL table columns into the file resource record fields expected by the domain layer
        :return a file resource record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'locator': self.locator,
                'mimetype': self.mimetype,
                }
