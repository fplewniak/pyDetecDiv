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


class FOVdao(Base):
    """
    DAO class for access to FOV records from the SQL database
    """
    __table__ = tables.list['FOV']

    def __init__(self, engine):
        self.engine = engine

    def get_records(self, where_clause):
        """
        A method to get from the SQL database, all FOV records verifying the specified where clause
        :param where_clause: the selection 'where clause' that can be specified using DAO classes or tables
        :type where_clause: a sqlachemy where clause defining the SQL selection query
        :return: a list of FOV records as dictionaries
        :rtype: list of dict
        """
        stmt = self.__table__.select(where_clause)
        result = self.engine.execute(stmt)
        return [FOVdao._create_record(fov) for fov in DataFrame(result.mappings()).to_dict('records')]

    @staticmethod
    def create_record(fov):
        """
        A private method creating a record dictionary from a fov row dictionary. This method is used to convert the SQL
        table columns into the FOV record fields expected by the domain layer
        :param fov: The dictionary created by the SQL database query and representing the data as it is stored in the
        database. This dictionary will be converted by this method into a dictionary record suitable for the domain
        layer
        :type fov: dict
        :return a FOV record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id': fov['id'],
                'name': fov['name'],
                'comments': fov['comments'],
                'shape': (fov['xsize'], fov['ysize'])
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


class ROIdao(Base):
    """
    DAO class for access to ROI records from the SQL database
    """
    __table__ = tables.list['ROI']

    def __init__(self, engine):
        self.engine = engine

    def get_records(self, where_clause):
        """
        A method to get from the SQL database, all ROI records verifying the specified where clause
        :param where_clause: the selection 'where clause' that can be specified using DAO classes or tables
        :type where_clause: a sqlachemy where clause defining the SQL selection query
        :return: a list of ROI records as dictionaries
        :rtype: list of dict
        """
        stmt = self.__table__.select(where_clause)
        result = self.engine.execute(stmt)
        return [ROIdao.create_record(roi) for roi in DataFrame(result.mappings()).to_dict('records')]

    @staticmethod
    def create_record(roi):
        """
        A private method creating a record dictionary from a roi row dictionary. This method is used to convert the SQL
        table columns into the ROI record fields expected by the domain layer
        :param roi: The dictionary created by the SQL database query and representing the data as it is stored in the
        database. This dictionary will be converted by this method into a dictionary record suitable for the domain
        layer
        :type roi: dict
        :return a ROI record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id': roi['id'],
                'name': roi['name'],
                'fov': roi['fov'],
                'top_left': (roi['x0'], roi['y0']),
                'bottom_right': (roi['x1'], roi['y1']),
                'shape': (roi['x1'] - roi['x0'] + 1, roi['y1'] - roi['y0'] + 1)
                }
