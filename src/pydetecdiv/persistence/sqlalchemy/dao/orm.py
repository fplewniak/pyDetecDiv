#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
Classes of DAO accessing data in SQL Tables, created by Table reflection.
These objects are responsible for providing the domain layer with lists of compatible records for the creation of
domain-specific objects.
"""
from sqlalchemy.orm import registry
from sqlalchemy.sql.expression import Insert, Update
from pandas import DataFrame
from pydetecdiv.persistence.sqlalchemy.dao.tables import Tables

mapper_registry = registry()
Base = mapper_registry.generate_base()
tables = Tables()


class DAO:
    """
    Data Access Object class defining methods common to all DAOs. Actual DAOs should inherit of this class first in
    order to inherit the __init__ method.
    """
    __table__ = None

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
        record = self.__class__._translate_record(rec)
        primary_key = self.session.execute(Insert(self.__class__).values(record)).inserted_primary_key[0]
        self.session.commit()
        return primary_key

    def update(self, rec):
        """
        Updates data in SQL database for the object corresponding to the record, which should contain the id of the
        modified object
        :param rec: the record representing the object
        :type rec: dict
        """
        id_ = rec['id']
        record = self.__class__._translate_record(rec)
        self.session.execute(Update(self.__class__, whereclause=self.__class__.id == id_).values(record))
        self.session.commit()

    @staticmethod
    def _translate_record(rec):
        """
        A private method to translate the data exchange record into a record suitable for insertion in the SQL database.
        This method in DAO, does not do anything and passes the record as is. It is supposed to be overridden by
        subclasses in order to actually translate the records accordingly
        :param rec: the record to be translated representing the object
        :type rec: dict
        :return: the translated record
        :rtype: dict
        """
        exclude = []
        translate = {}
        return DAO.translate_record(rec, exclude, translate)

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

    @staticmethod
    def create_record(rec):
        """
        Template for method converting a DAO row dictionary into a DSO record
        :param rec: the DAO record to convert into a DSO record
        :type rec: dict
        :return: a DSO record
        :rtype: dict
        """
        raise NotImplementedError('Call to a create_record(rec) method that is not implemented')


class FOVdao(DAO, Base):
    """
    DAO class for access to FOV records from the SQL database
    """
    __table__ = tables.list['FOV']

    @staticmethod
    def _translate_record(rec):
        """
        A private method to translate the data exchange record into a record suitable for insertion in the SQL database
        :param rec: the record to be translated representing the FOV
        :type rec: dict
        :return: the translated record
        :rtype: dict
        """
        exclude = ['id', 'top_left', 'bottom_right']
        translate = {'size': ('xsize', 'ysize'), }
        return DAO.translate_record(rec, exclude, translate)

    @staticmethod
    def create_record(rec):
        """
        A method creating a DAO record dictionary from a fov row dictionary. This method is used to convert the SQL
        table columns into the FOV record fields expected by the domain layer
        :param rec: The dictionary created by the SQL database query and representing the data as it is stored in the
        database. This dictionary will be converted by this method into a dictionary record suitable for the domain
        layer
        :type rec: dict
        :return a FOV record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id': rec['id'],
                'name': rec['name'],
                'comments': rec['comments'],
                'top_left': (0, 0),
                'bottom_right': (rec['xsize'] - 1, rec['ysize'] - 1),
                'size': (rec['xsize'], rec['ysize']),
                }

    def roi_list(self, fov_id):
        """
        A method returning the list of ROIs whose parent FOV has id == fov_id
        :param fov_id: the id of the FOV
        :type fov_id: int
        :return: a list of ROI records with parent FOV id == fov_id
        :rtype: list
        """
        roi_dao = ROIdao(self.session)
        return roi_dao.get_records(tables.list['ROI'].c.fov == fov_id)


class ROIdao(DAO, Base):
    """
    DAO class for access to ROI records from the SQL database
    """
    __table__ = tables.list['ROI']

    @staticmethod
    def _translate_record(rec):
        """
        A private method to translate the data exchange record into a record suitable for insertion in the SQL database
        :param rec: the record to be translated representing the ROI
        :type rec: dict
        :return: the translated record
        :rtype: dict
        """
        exclude = ['id', 'size', ]
        translate = {'top_left': ('x0', 'y0'), 'bottom_right': ('x1', 'y1')}
        return DAO.translate_record(rec, exclude, translate)

    @staticmethod
    def create_record(rec):
        """
        A method creating a record dictionary from a roi row dictionary. This method is used to convert the SQL
        table columns into the ROI record fields expected by the domain layer
        :param rec: The dictionary created by the SQL database query and representing the data as it is stored in the
        database. This dictionary will be converted by this method into a dictionary record suitable for the domain
        layer
        :type rec: dict
        :return a ROI record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id': rec['id'],
                'name': rec['name'],
                'fov': rec['fov'],
                'top_left': (rec['x0'], rec['y0']),
                'bottom_right': (rec['x1'], rec['y1']),
                'size': (rec['x1'] - rec['x0'] + 1, rec['y1'] - rec['y0'] + 1)
                }
