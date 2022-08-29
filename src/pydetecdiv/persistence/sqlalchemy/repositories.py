import pathlib
import re
from importlib.resources import files as resource_files
import sqlalchemy
from sqlalchemy.orm import Session
import pydetecdiv
from pydetecdiv.persistence.repository import ShallowDb

class _ShallowSQL(ShallowDb):
    """
    A generic shallow SQL persistence used to provide the common methods for SQL databases. DBMS-specific methods should be
    implemented in subclasses of this one.
    """

    def __init__(self, dbname):
        self.name = dbname
        self.engine = None

    def executescript(self, script: pathlib.Path):
        """
        Reads a file containing several SQL statements in a free format.
        :param script: the path of the SQL script to be executed
        """
        try:
            with Session(self.engine, future=True) as session:
                with open(script, 'r') as f:
                    statements = re.split(r';\s*$', f.read(), flags=re.MULTILINE)
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
            self.executescript(resource_files(pydetecdiv.database).joinpath('CreateTables.sql'))

    def close(self):
        """
        Close the current connexion.
        """
        self.engine.dispose()

    def get_objects(self, class_: type = None, query: list=None):
        """

        :param class_:
        :param query:
        """
        stmt = sqlalchemy.select(class_.dao)
        if query is not None:
            for q in query:
                stmt = stmt.where(q)
        with Session(self.engine) as session:
            result = session.execute(stmt)
            return [class_(row) for row in result]


class _ShallowSQLite3(_ShallowSQL):
    """
    A concrete shallow SQLite3 persistence inheriting _ShallowSQL and implementing SQLite3-specific engine.
    """

    def __init__(self, dbname):
        super().__init__(dbname)
        self.engine = sqlalchemy.create_engine(f'sqlite+pysqlite:///{self.name}')
        super().create()
