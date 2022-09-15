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
from sqlalchemy.schema import Index
from sqlalchemy import text
from sqlalchemy.orm import relationship
from pandas import DataFrame

# from pydetecdiv.persistence.sqlalchemy.dao.tables import Tables
#
mapper_registry = registry()
Base = mapper_registry.generate_base()


class DAO:
    """
    Data Access Object class defining methods common to all DAOs. This class is not meant to be used directly.
    Actual DAOs should inherit of this class first in order to inherit the __init__ method.
    """
    __table__ = None
    exclude = []
    translate = {}

    def __init__(self, session):
        self.session = session

    def insert(self, rec):
        """
        Inserts data in SQL database for a newly created object
        :param rec: the record representing the object
        :type rec: dict
        :return: the primary key of the newly created object
        :rtype: int
        """
        record = self.translate_record(rec, self.exclude, self.translate)
        primary_key = self.session.execute(Insert(self.__class__).values(record)).inserted_primary_key[0]
        self.session.commit()
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
        id_ = rec['id']
        record = self.translate_record(rec, self.exclude, self.translate)
        self.session.execute(Update(self.__table__, whereclause=self.__table__.c.id == id_).values(record))
        self.session.commit()
        return id_

    def get_records(self, where_clause):
        """
        A method to get from the SQL database, all records verifying the specified where clause
        Example of use:
        roi_records = roidao.get_records((ROIdao.fov == FOVdao.id) & (FOVdao.name == 'fov1')) retrieves all ROI records
        associated with FOV whose name is 'fov1'

        :param where_clause: the selection 'where clause' that can be specified using DAO classes or tables
        :type where_clause: a sqlachemy where clause defining the SQL selection query
        :return: a list of records as dictionaries
        :rtype: list of dict
        """
        stmt = self.__table__.select(where_clause)
        result = self.session.execute(stmt)
        return [self.__class__.create_record(rec) for rec in DataFrame(result.mappings()).to_dict('records')]

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
                rec[translate[key][0]], rec[translate[key][1]] = record[key]
            else:
                rec[key] = record[key]
        return rec

    def create_record(self):
        """
        Template for method converting a DAO row dictionary into a DSO record
        :return: a DSO record
        :rtype: dict
        """
        raise NotImplementedError('Call to a create_record(rec) method that is not implemented')


class FOVdao(DAO, Base):
    """
    DAO class for access to FOV records from the SQL database
    """
    __tablename__ = 'FOV'
    exclude = ['id', 'top_left', 'bottom_right']
    translate = {'size': ('xsize', 'ysize'), }

    id = Column(Integer, primary_key=True, autoincrement='auto')
    name = Column(String, unique=True)
    comments = Column(String)
    xsize = Column(Integer, nullable=False, server_default=text('1000'))
    ysize = Column(Integer, nullable=False, server_default=text('1000'))

    roi_list_ = relationship('ROIdao')

    def create_record(self):
        """
        A method creating a DAO record dictionary from a fov row dictionary. This method is used to convert the SQL
        table columns into the FOV record fields expected by the domain layer
        :return a FOV record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        # rec = rec['FOVdao']
        return {'id': self.id,
                'name': self.name,
                'comments': self.comments,
                'top_left': (0, 0),
                'bottom_right': (self.xsize - 1, self.ysize - 1),
                'size': (self.xsize, self.ysize),
                }

    def roi_list(self, fov_id):
        """
        A method returning the list of ROIs whose parent FOV has id == fov_id
        :param fov_id: the id of the FOV
        :type fov_id: int
        :return: a list of ROI records with parent FOV id == fov_id
        :rtype: list
        """
        return [roi.create_record()
                for roi in self.session.query(FOVdao)
                .options(joinedload(FOVdao.roi_list_))
                .filter(FOVdao.id == fov_id)
                .first().roi_list_]


class ROIdao(DAO, Base):
    """
    DAO class for access to ROI records from the SQL database
    """
    __tablename__ = 'ROI'
    exclude = ['id', 'size', ]
    translate = {'top_left': ('x0', 'y0'), 'bottom_right': ('x1', 'y1')}

    id = Column(Integer, primary_key=True, autoincrement='auto')
    name = Column(String, unique=True)
    fov = Column(Integer, ForeignKey('FOV.id'), nullable=False, index=True)
    x0 = Column(Integer, nullable=False, server_default=text('0'))
    y0 = Column(Integer, nullable=False, server_default=text('-1'))
    x1 = Column(Integer, nullable=False, server_default=text('0'))
    y1 = Column(Integer, nullable=False, server_default=text('-1'))

    def create_record(self):
        """
        A method creating a record dictionary from a roi row dictionary. This method is used to convert the SQL
        table columns into the ROI record fields expected by the domain layer
        :return a ROI record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id': self.id,
                'name': self.name,
                'fov': self.fov,
                'top_left': (self.x0, self.y0),
                'bottom_right': (self.x1, self.y1),
                'size': (self.x1 - self.x0 + 1, self.y1 - self.y0 + 1)
                }
