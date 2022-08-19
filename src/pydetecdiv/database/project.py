#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Project database management
"""
import abc
import sqlite3
from importlib.resources import files as resource_files
import pydetecdiv
from pydetecdiv.settings import get_config_value


class ShallowDb(abc.ABC):
    """
    Abstract class used as an interface to encapsulate project database access
    """

    @abc.abstractmethod
    def close(self):
        """
        Abstract method enforcing the implementation of a close() method in all shallow database connectors
        """


def open_project(dbname: str = None) -> ShallowDb:
    """
    A function to open a shallow database from its name. The type of database is defined in the [project] sections of
    the configuration file settings.ini
    :param dbname: the database name
    :return: a shallowDb abstract connector encapsulating the concrete connectors
    """
    dbms = get_config_value('project', 'dbms')
    if dbms == 'SQLite3':
        dbname = dbname if dbname is not None else get_config_value('project.sqlite', 'database')
        db = _ShallowSQLite3(dbname)
    else:
        print(f'{dbms} is not implemented')
    return db


class _ShallowSQL(ShallowDb):
    """
    A generic shallow SQL database used to provide the common methods for SQL databases. DBMS-specific methods should be
    implemented in subclasses of this one.
    """

    def __init__(self, dbname):
        self.name = dbname
        self.con = None

    def create(self):
        """
        Reads the SQL script 'CreateTables.sql' resource file and passes it to the executescript method of the subclass
        for execution according to the DBMS-specific functionalities.
        """
        self.executescript(resource_files(pydetecdiv.database).joinpath('CreateTables.sql').read_text())

    def close(self):
        """
        Close the current connexion.
        """
        self.con.close()


class _ShallowSQLite3(_ShallowSQL):
    """
    A concrete shallow SQLite3 database inheriting _ShallowSQL and implementing SQLite3-specific engine.
    """

    def __init__(self, dbname):
        super().__init__(dbname)
        self.con = sqlite3.connect(self.name)
        self.cursor = self.con.cursor()
        super().create()

    def executescript(self, script: str):
        """
        Execute a SQL script and handling exceptions
        :param script:
        """
        try:
            self.cursor.executescript(script)
        except sqlite3.OperationalError as exc:
            print(exc)
