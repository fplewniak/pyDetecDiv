#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Concrete Repositories using a SQL database with the sqlalchemy toolkit
"""
import re
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import Delete
from sqlalchemy.pool import NullPool
from pandas import DataFrame
from pydetecdiv.persistence.repository import ShallowDb
from pydetecdiv.persistence.sqlalchemy.orm.main import mapper_registry
from pydetecdiv.persistence.sqlalchemy.orm.dao import dso_dao_mapping as dao
from pydetecdiv.persistence.sqlalchemy.orm.associations import Linker


class _ShallowSQL(ShallowDb):
    """
    A generic shallow SQL persistence used to provide the common methods for SQL databases. DBMS-specific methods should
    be implemented in subclasses of this one.
    """

    def __init__(self, dbname):
        self.name = dbname
        self.engine = sqlalchemy.create_engine('sqlite+pysqlite://', poolclass=NullPool)
        self.session_maker = sessionmaker(self.engine)

    def executescript(self, script):
        """
        Reads a string containing several SQL statements in a free format
        :param script: the string representing the SQL script to be executed
        :type script: str
        """
        try:
            statements = re.split(r';\s*$', script, flags=re.MULTILINE)
            with self.session_maker() as session:
                for statement in statements:
                    if statement:
                        session.execute(sqlalchemy.text(statement))
        except sqlalchemy.exc.OperationalError as exc:
            print(exc)

    def create(self):
        """
        Gets SqlAlchemy classes defining the project database schema and creates the database if it does not exist.
        """
        if not self.engine.table_names():
            mapper_registry.metadata.create_all(self.engine)

    def close(self):
        """
        Close the current connexion
        """
        self.engine.dispose()

    def save_object(self, class_name, record):
        """
        Save the object represented by the record
        :param class_name: the class name of the object to save into SQL database
        :type class_name: str
        :param record: the record representing the object
        :type record: dict
        :return: the id of the created or updated object
        :rtype: int
        """
        if record['id_'] is None:
            return dao[class_name](self.session_maker).insert(record)
        return dao[class_name](self.session_maker).update(record)

    def delete_object(self, class_name, id_):
        """
        Delete an object of class name = class_name with id = id_
        :param class_name: the class name of the object to delete
        :type class_name: str
        :param id_: the id of the object to delete
        :type id_: int
        """
        with self.session_maker() as session:
            session.execute(Delete(dao[class_name], whereclause=dao[class_name].id_ == id_))
            session.commit()

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
        with self.session_maker() as session:
            dao_list = session.query(dao[class_name])
            if query is not None:
                for q in query:
                    dao_list = dao_list.where(q)
        return [obj.record for obj in dao_list]

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
        with self.session_maker() as session:
            return session.get(dao[class_name], id_).record

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
        with self.session_maker() as session:
            if id_list is not None:
                dao_list = session.query(dao[class_name]).where(dao[class_name].id_.in_(id_list))
            else:
                dao_list = session.query(dao[class_name])
        return [obj.record for obj in dao_list]

    def get_linked_records(self, class_name, parent_class_name, parent_id):
        """
        A method returning the list of records for all objects of class defined by class_name that are linked to object
        of class parent_class_name with id_ = parent_id
        :param class_name: the class name of the objects to retrieve records for
        :type class_name: str
        :param parent_class_name: the class nae of the parent object
        :type parent_class_name: str
        :param parent_id: the id of the parent object
        :type parent_id: int
        :return: a list of records
        :rtype: list of dict
        """
        linked_records = []
        if class_name == 'ImageData':
            if parent_class_name in ['FileResource', 'FOV']:
                linked_records = dao[parent_class_name](self.session_maker).image_data(parent_id)
        if class_name == 'FOV':
            if parent_class_name in ['ImageData']:
                linked_records = dao[parent_class_name](self.session_maker).fov_list(parent_id)
            if parent_class_name in ['ROI']:
                linked_records = [self.get_record(class_name, self.get_record(parent_class_name, parent_id)['id_'])]
        if class_name == 'ROI':
            if parent_class_name in ['FOV']:
                linked_records = dao[parent_class_name](self.session_maker).roi_list(parent_id)
        return linked_records

    def link(self, class1_name, id_1, class2_name, id_2):
        """
        Create a link between two domain-specific objects. There must be a direct link defined in Linker class,
        otherwise, the link cannot be created.
        :param class1_name: the class name of the first object to link
        :type class1_name: str
        :param id_1: the id of the first object to link
        :type id_1: int
        :param class2_name: the class name of the second object to link
        :type class2_name: str
        :param id_2: the id of the second object to link
        :type id_2: int
        """
        with self.session_maker() as session:
            obj1 = session.get(dao[class1_name], id_1)
            obj2 = session.get(dao[class2_name], id_2)
            Linker.link(obj1, obj2)
            session.commit()



class ShallowSQLite3(_ShallowSQL):
    """
    A concrete shallow SQLite3 persistence inheriting _ShallowSQL and implementing SQLite3-specific engine
    """

    def __init__(self, dbname=None):
        super().__init__(dbname)
        self.engine = sqlalchemy.create_engine(f'sqlite+pysqlite:///{self.name}', poolclass=NullPool)
        self.session_maker = sessionmaker(self.engine, future=True)
        super().create()
