import pathlib
import re
from importlib.resources import files as resource_files
import sqlalchemy
from sqlalchemy.orm import Session
import pydetecdiv
from pydetecdiv.persistence.repository import ShallowDb
from pydetecdiv.persistence.sqlalchemy.dao.orm import FOV_DAO, ROI_DAO


class _ShallowSQL(ShallowDb):
    """
    A generic shallow SQL persistence used to provide the common methods for SQL databases. DBMS-specific methods should be
    implemented in subclasses of this one.
    """
    class_mapping = {'FOV': FOV_DAO,
                     'ROI': ROI_DAO,
                     }

    def __init__(self, dbname):
        self.name = dbname
        self.engine = None

    def executescript(self, script: str):
        """
        Reads a string containing several SQL statements in a free format.
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
        Reads the SQL script 'CreateTables.sql' resource file and passes it to the executescript method of the subclass
        for execution according to the DBMS-specific functionalities.
        """
        if not self.engine.table_names():
            with open(resource_files(pydetecdiv.persistence).joinpath('CreateTables.sql'), 'r') as f:
                self.executescript(f.read())

    def close(self):
        """
        Close the current connexion.
        """
        self.engine.dispose()

    def get_objects(self, class_: type = None, query: list = None):
        """

        :param class_:
        :param query:
        """
        stmt = sqlalchemy.select(self.class_mapping[class_.__name__])
        if query is not None:
            for q in query:
                stmt = stmt.where(q)
        with Session(self.engine) as session:
            result = session.execute(stmt)
            return [class_({"name": row['FOV_DAO'].name}) for row in result]


class _ShallowSQLite3(_ShallowSQL):
    """
    A concrete shallow SQLite3 persistence inheriting _ShallowSQL and implementing SQLite3-specific engine.
    """

    def __init__(self, dbname):
        super().__init__(dbname)
        self.engine = sqlalchemy.create_engine(f'sqlite+pysqlite:///{self.name}')
        super().create()
