#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
Classes of DAO accessing data in SQL Tables, created by Table reflection.
These objects are responsible for providing the domain layer with lists of compatible records for the creation of
domain-specific objects.
"""
from sqlalchemy.orm import registry, joinedload
from sqlalchemy.sql.expression import Insert, Update
from sqlalchemy import Column, Integer, String, Time, DateTime, ForeignKey
from sqlalchemy import text
from sqlalchemy.orm import relationship

mapper_registry = registry()
Base = mapper_registry.generate_base()


class DAO:
    """
    Data Access Object class defining methods common to all DAOs. This class is not meant to be used directly.
    Actual DAOs should inherit of this class first in order to inherit the __init__ method.
    """
    id_ = None
    exclude = []
    translate = {}

    def __init__(self, session_maker):
        self.session_maker = session_maker

    def insert(self, rec):
        """
        Inserts data in SQL database for a newly created object
        :param rec: the record representing the object
        :type rec: dict
        :return: the primary key of the newly created object
        :rtype: int
        """
        record = self.translate_record(rec, self.exclude, self.translate)
        with self.session_maker() as session:
            primary_key = session.execute(Insert(self.__class__).values(record)).inserted_primary_key[0]
            session.commit()
        return primary_key

    def update(self, rec):
        """
        Updates data in SQL database for the object corresponding to the record, which should contain the id of the
        modified object
        :param rec: the record representing the object
        :type rec: dict
        :return: the primary key of the updated object
        :rtype: int
        """
        id_ = rec['id_']
        record = self.translate_record(rec, self.exclude, self.translate)
        with self.session_maker() as session:
            session.execute(Update(self.__class__, whereclause=self.__class__.id_ == id_).values(record))
            session.commit()
        return id_

    def get_records(self, where_clause):
        """
        A method to get from the SQL database, all records verifying the specified where clause
        Example of use:
        roi_records = roidao.get_records((ROIdao.fov == FOVdao.id_) & (FOVdao.name == 'fov1')) retrieves all ROI records
        associated with FOV whose name is 'fov1'

        :param where_clause: the selection 'where clause' that can be specified using DAO classes or tables
        :type where_clause: a sqlachemy where clause defining the SQL selection query
        :return: a list of records as dictionaries
        :rtype: list of dict
        """
        with self.session_maker() as session:
            dao_list = session.query(self.__class__).where(where_clause)
        return [obj.record for obj in dao_list]

    @staticmethod
    def translate_record(record, exclude, translate):
        """
        The actual translation engine reading the record fields and translating or excluding them if they occur in the
        translate or exclude variables
        :param record: the record to be translated
        :type record: dict
        :param exclude: a list of fields that must not be passed to the SQL engine
        :type exclude: list
        :param translate: a list of fields that must be translated and the corresponding columns
        :type translate: dict
        :return: the translated record
        :rtype: dict
        """
        rec = {}
        for key in record:
            if key in exclude:
                continue
            if key in translate:
                if isinstance(translate[key], (list, tuple)):
                    for translated_key, value in zip(translate[key], record[key]):
                        rec[translated_key] = value
                else:
                    rec[translate[key]] = record[key]
            else:
                rec[key] = record[key]
        return rec

    @property
    def record(self):
        """
        Template for method converting a DAO row dictionary into a DSO record
        :return: a DSO record
        :rtype: dict
        """
        raise NotImplementedError('Call to a record() method that is not implemented')


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

    image_data_list = relationship("FovData", back_populates='one_fov', lazy='joined')

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
            image_data = [association.image_data.record
                          for association in session.query(FOVdao)
                          .options(joinedload(FOVdao.image_data_list))
                          .filter(FOVdao.id_ == fov_id)
                          .first().image_data_list]
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


class ROIdao(DAO, Base):
    """
    DAO class for access to ROI records from the SQL database
    """
    __tablename__ = 'ROI'
    exclude = ['id_', 'size', ]
    translate = {'top_left': ('x0_', 'y0_'), 'bottom_right': ('x1_', 'y1_')}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    name = Column(String, unique=True, nullable=False)
    fov = Column(Integer, ForeignKey('FOV.id_'), nullable=False, index=True)
    x0_ = Column(Integer, nullable=False, server_default=text('0'))
    y0_ = Column(Integer, nullable=False, server_default=text('-1'))
    x1_ = Column(Integer, nullable=False, server_default=text('0'))
    y1_ = Column(Integer, nullable=False, server_default=text('-1'))

    @property
    def record(self):
        """
        A method creating a record dictionary from a roi row dictionary. This method is used to convert the SQL
        table columns into the ROI record fields expected by the domain layer
        :return a ROI record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'name': self.name,
                'fov': self.fov,
                'top_left': (self.x0_, self.y0_),
                'bottom_right': (self.x1_, self.y1_),
                'size': (self.x1_ - self.x0_ + 1, self.y1_ - self.y0_ + 1)
                }


class ImageDataDao(DAO, Base):
    """
    DAO class for access to ImageData records from the SQL database
    """
    __tablename__ = 'ImageData'
    exclude = ['id_', ]
    translate = {'top_left': ('x0_', 'y0_'), 'bottom_right': ('x1_', 'y1_'), 'file_resource': 'resource'}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    name = Column(String, unique=True, )
    channel = Column(Integer, nullable=False, )
    x0_ = Column(Integer, nullable=False, server_default=text('0'))
    y0_ = Column(Integer, nullable=False, server_default=text('-1'))
    x1_ = Column(Integer, nullable=False, server_default=text('0'))
    y1_ = Column(Integer, nullable=False, server_default=text('-1'))
    stacks = Column(Integer, nullable=False, server_default=text('1'))
    frames = Column(Integer, nullable=False, server_default=text('1'))
    interval = Column(Time, )
    orderdims = Column(String, nullable=False, server_default=text('xyzct'))
    resource = Column(Integer, ForeignKey('FileResource.id_'), nullable=False, index=True)
    path = Column(String, )
    mimetype = Column(String)

    fov_list_ = relationship("FovData", back_populates='image_data', lazy='joined')

    def fov_list(self, image_data_id):
        """
        A method returning the list of FOV object records linked to ImageData with id_ == image_data_id
        :param image_data_id: the id of the Image data
        :type image_data_id: int
        :return: a list of FOV records linked to ImageData with id_ == image_data_id
        :rtype: list
        """
        with self.session_maker() as session:
            fov_list = [association.one_fov.record
                        for association in session.query(ImageDataDao)
                        .options(joinedload(ImageDataDao.fov_list_))
                        .filter(ImageDataDao.id_ == image_data_id)
                        .first().fov_list_]
        return fov_list

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
                'stacks': self.stacks,
                'frames': self.frames,
                'interval': self.interval,
                'orderdims': self.orderdims,
                'file_resource': self.resource,
                'path': self.path,
                'mimetype': self.mimetype,
                }


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


class FovData(DAO, Base):
    """
    Association many to many between FOV and Image data
    """
    __tablename__ = "FOVdata"
    left_id = Column(ForeignKey("FOV.id_"), primary_key=True)
    right_id = Column(ForeignKey("ImageData.id_"), primary_key=True)
    one_fov = relationship("FOVdao", back_populates='image_data_list', lazy='joined')
    image_data = relationship("ImageDataDao", back_populates='fov_list_', lazy='joined')

    @property
    def record(self):
        """
        A method creating a record dictionary from a FovData row dictionary.
        :return a FovData record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {
            'fov': self.left_id,
            'image_data': self.right_id,
        }
