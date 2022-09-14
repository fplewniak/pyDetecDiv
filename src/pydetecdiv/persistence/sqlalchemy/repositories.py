#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Concrete Repositories using a SQL database with the sqlalchemy toolkit
"""
import re
import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import Delete
from sqlalchemy.pool import SingletonThreadPool
from pandas import DataFrame
from pydetecdiv.persistence.repository import ShallowDb
from pydetecdiv.persistence.sqlalchemy.dao.tables import Tables
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
        self.session = None
        self.tables = None

    def executescript(self, script):
        """
        Reads a string containing several SQL statements in a free format
        :param script: the string representing the SQL script to be executed
        :type script: str
        """
        try:
            statements = re.split(r';\s*$', script, flags=re.MULTILINE)
            for statement in statements:
                if statement:
                    self.session.execute(sqlalchemy.text(statement))
            self.session.commit()
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

    def save(self, class_name, record):
        """
        Save the object represented by the record
        :param class_name: the class name of the object to save into SQL database
        :type class_name: str
        :param record: the record representing the object
        :type record: dict
        """
        if record['id'] is None:
            return self.dao[class_name](self.session).insert(record)
        return self.dao[class_name](self.session).update(record)

    def delete(self, class_name, id_):
        self.session.execute(Delete(self.dao[class_name], whereclause=self.dao[class_name].id == id_))
        self.session.commit()

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
        result = self.session.execute(stmt)
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

    # def _get_records_using_dao(self, class_name=None, query=None):
    #     return [self.dao[class_name](self.session).get_records(where_clause) for where_clause in query]

    def get_dataframe(self, class_name, id_list=None):
        """
        Get a DataFrame containing the list of all domain objects of a given class in the current project
        :param class_name: the class name of the objects whose list will be returned
        :type class_name: str
        :param id_list: the list of ids of objects to retrieve
        :type id_list: a list of int
        :return: a DataFrame containing the list of objects
        :rtype: DataFrame containing the records representing the requested domain-specific objects
        """
        return DataFrame(self.get_records(class_name, id_list))

    def get_record(self, class_name, id_=None):
        """
        A method returning an object record of a given class from its id
        :param class_name: the class name of object to get the record of
        :type class_name: str
        :param id_: the id of the requested object
        :type id_: int
        :return: the object record
        :rtype: dict (record)
        """
        return self._get_records(class_name, [self.tables.list[class_name].c.id == id_])[0]

    def get_records(self, class_name, id_list=None):
        """
        A method returning the list of all object records of a given class or select those whose id is in id_list
        :param class_name: the class name of objects to get records of
        :type class_name: str
        :param id_list: the list of ids of objects to retrieve
        :type id_list: a list of int
        :return: a list of records
        :rtype: list of dictionaries (records)
        """
        return self._get_records(class_name) if id_list is None else [self.get_record(class_name, id_) for id_ in
                                                                      id_list]

    def get_roi_list_in_fov(self, fov_id):
        """
        A method returning the list of records for all ROI in the FOV with id == fov_id
        :param fov_id: the id of the FOV
        :type fov_id: int
        :return: a list of ROIs whose parent if the FOV with id == fov_id
        :rtype: list of dictionaries (records)
        """
        fov_dao = FOVdao(self.session)
        return fov_dao.roi_list(fov_id)


class ShallowSQLite3(_ShallowSQL):
    """
    A concrete shallow SQLite3 persistence inheriting _ShallowSQL and implementing SQLite3-specific engine
    """

    def __init__(self, dbname=None):
        super().__init__(dbname)
        self.engine = sqlalchemy.create_engine(f'sqlite+pysqlite:///{self.name}', poolclass=SingletonThreadPool)
        # self.engine = sqlalchemy.create_engine(f'sqlite+pysqlite:///{self.name}', poolclass=SingletonThreadPool,
        #                                        echo=True, echo_pool='debug')
        self.session = Session(self.engine, future=True)
        self.session.begin()
        super().create()
