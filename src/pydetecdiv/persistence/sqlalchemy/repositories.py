#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Concrete Repositories using a SQL database with the sqlalchemy toolkit
"""
import re
import sqlalchemy
from sqlalchemy.orm import Session
from pandas import DataFrame
from pandas import read_sql
from pydetecdiv.persistence.repository import ShallowDb
from pydetecdiv.persistence.sqlalchemy.dao.tables import Tables
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.persistence.sqlalchemy.dao.orm import FOVdao, ROIdao


class _ShallowSQL(ShallowDb):
    """
    A generic shallow SQL persistence used to provide the common methods for SQL databases. DBMS-specific methods should
    be implemented in subclasses of this one.
    The dao dictionary maps the correspondence between domain-specific class names and DAO classes
    """
    dao = {
        'FOV': FOVdao,
        'ROI': ROIdao
    }

    def __init__(self, dbname):
        self.name = dbname
        self.engine = None
        self.tables = None

    def executescript(self, script):
        """
        Reads a string containing several SQL statements in a free format
        :param script: the string representing the SQL script to be executed
        :type script: str
        """
        try:
            with Session(self.engine, future=True) as session:
                statements = re.split(r';\s*$', script, flags=re.MULTILINE)
                for statement in statements:
                    if statement:
                        session.execute(sqlalchemy.text(statement))
                session.commit()
        except sqlalchemy.exc.OperationalError as exc:
            print(exc)

    def create(self):
        """
        Gets SqlAlchemy classes defining the project database schema and creates the database if it does not exist.
        """
        self.tables = Tables()
        if not self.engine.table_names():
            self.tables.create(self.engine)

    def close(self):
        """
        Close the current connexion
        """
        self.engine.dispose()

    def _get_raw_objects_df(self, selection=None, query=None):
        """
        Get objects specified by the table list and satisfying the conditions defined by the list of queries. This
        method is not supposed to be called from outside this class
        Example of usage to retrieve FOVs and the name of their associated ROIs:
        roi_table = self.tables.list['ROI']
        fov_table = self.tables.list['FOV']
        self._get_objects([fov_table, roi_table.c.name], query=[fov_table.c.id == roi_table.c.fov])

        :param selection: a list of tables or columns thereof to select rows from. Can be defined by a list containing
         t.list[table_name] or t.columns(table_name).column_name or any combination thereof
        :param query: a list of queries on tables
        :type selection: list of tables of columns to select from the database
        :type query: a list of sqlalchemy where clauses
        :return a DataFrame containing the requested objects
        :rtype: DataFrame
        """
        stmt = sqlalchemy.select(selection)
        if query is not None:
            for q in query:
                stmt = stmt.where(q)
        result = self.engine.execute(stmt)
        return DataFrame(result.mappings())

    def _get_records(self, class_name=None, query=None):
        """
        A private method returning the list of all object records of a given class specified by its name and verifying a
        query built from a list of where clauses
        :param class_name: the name of the class of objects to get records of
        :param query: a list of sqlalchemy where clauses defining the selection SQL query
        :type class_name: str
        :type query: list of where clauses
        :return: a list of records
        :rtype: list of dictionaries (records)
        """
        selection = self.tables.list[class_name]
        return [self.dao[class_name].create_record(rec) for rec in
                self._get_raw_objects_df(selection, query).to_dict('records')]

    def get_records(self, class_=DomainSpecificObject):
        """
        A method returning the list of all object records of a given class
        :param class_: the class of objects to get records of
        :type class_: class
        :return: a list of records
        :rtype: list of dictionaries (records)
        """
        return self._get_records(class_.__name__)

    def get_roi_list_in_fov(self, fov_id):
        """
        A method returning the list of records for all ROI in the FOV with id == fov_id
        :param fov_id: the id of the FOV
        :type fov_id: int
        :return: a list of ROIs whose parent if the FOV with id == fov_id
        :rtype: list of dictionaries (records)
        """
        fov_dao = FOVdao(self.engine)
        return fov_dao.roi_list(fov_id)


class ShallowSQLite3(_ShallowSQL):
    """
    A concrete shallow SQLite3 persistence inheriting _ShallowSQL and implementing SQLite3-specific engine
    """

    def __init__(self, dbname=None):
        super().__init__(dbname)
        self.engine = sqlalchemy.create_engine(f'sqlite+pysqlite:///{self.name}')
        super().create()
