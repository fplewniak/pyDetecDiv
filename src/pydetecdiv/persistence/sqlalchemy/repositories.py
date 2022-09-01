#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Concrete Repositories using a SQL database with the sqlalchemy toolkit
"""
import re
import sqlalchemy
from sqlalchemy.orm import Session
from pandas import DataFrame
from pydetecdiv.persistence.repository import ShallowDb
from pydetecdiv.persistence.sqlalchemy.dao.tables import Tables


class _ShallowSQL(ShallowDb):
    """
    A generic shallow SQL persistence used to provide the common methods for SQL databases. DBMS-specific methods should
    be implemented in subclasses of this one
    """

    def __init__(self, dbname):
        self.name = dbname
        self.engine = None
        self.tables = None

    def executescript(self, script: str):
        """
        Reads a string containing several SQL statements in a free format
        :param script: the string representing the SQL script to be executed
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

    def _get_objects(self, classes: list = None, query: list = None) -> DataFrame:
        """
        Get objects specified by the table list and satisfying the conditions defined by the list of queries. This
        method is not supposed to be called from outside this class
        :param classes: a list of tables or columns thereof to select rows from. Can be defined by t.list[table_name] or
        t.columns(table_name).column_name or any combination thereof
        :param query: a list of queries on tables
        :return a list of dictionaries representing the requested objects obtained from sqlalchemy.engine.RowMapping
        objects returned by results.mappings()
        """
        stmt = sqlalchemy.select(classes)
        if query is not None:
            for q in query:
                stmt = stmt.where(q)
        with Session(self.engine) as session:
            results = session.execute(stmt)
            return DataFrame(results.mappings())

    def get_object_list(self, class_name: str = None, as_list: bool = True):
        """
        Return a list of objects of a given class specified by its name, which should be also the name of the
        corresponding sqlalchemy Table
        :param class_name: the class name
        :param as_list: if True, returns a list of dictionaries else returns a DataFrame
        :return: a DataFrame or a list of dictionaries containing the data for the requested objects
        """
        if as_list:
            object_list = self._get_objects(self.tables.list[class_name]).to_dict(orient='records')
        else:
            object_list = self._get_objects(self.tables.list[class_name])
        return object_list


class ShallowSQLite3(_ShallowSQL):
    """
    A concrete shallow SQLite3 persistence inheriting _ShallowSQL and implementing SQLite3-specific engine
    """

    def __init__(self, dbname: str = None) -> ShallowDb:
        super().__init__(dbname)
        self.engine = sqlalchemy.create_engine(f'sqlite+pysqlite:///{self.name}')
        super().create()
