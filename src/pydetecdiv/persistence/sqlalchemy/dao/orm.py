#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
Classes of DAO accessing data in SQL Tables, created by Table reflection.
These objects are responsible for providing the domain layer with lists of compatible records for the creation of
domain-specific objects.
"""
from sqlalchemy.orm import registry
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

    def __init__(self, engine):
        self.engine = engine

    def get_records(self, where_clause):
        """
        A method to get from the SQL database, all records verifying the specified where clause
        :param where_clause: the selection 'where clause' that can be specified using DAO classes or tables
        :type where_clause: a sqlachemy where clause defining the SQL selection query
        :return: a list of records as dictionaries
        :rtype: list of dict
        """
        stmt = self.__table__.select(where_clause)
        result = self.engine.execute(stmt)
        return [self.__class__.create_record(rec) for rec in DataFrame(result.mappings()).to_dict('records')]

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
        roi_dao = ROIdao(self.engine)
        return roi_dao.get_records(tables.list['ROI'].c.fov == fov_id)


class ROIdao(DAO, Base):
    """
    DAO class for access to ROI records from the SQL database
    """
    __table__ = tables.list['ROI']

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
