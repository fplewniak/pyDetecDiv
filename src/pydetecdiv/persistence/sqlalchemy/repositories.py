#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Concrete Repositories using a SQL database with the sqlalchemy toolkit
"""
import re
import sqlalchemy
from sqlalchemy.orm import Session
from pydetecdiv.persistence.repository import ShallowDb
from pydetecdiv.persistence.sqlalchemy.dao.orm import FOV_DAO, ROI_DAO
from pydetecdiv.persistence.sqlalchemy.dao.tables import Tables


class _ShallowSQL(ShallowDb):
    """
    A generic shallow SQL persistence used to provide the common methods for SQL databases. DBMS-specific methods should
    be implemented in subclasses of this one.
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
        Gets SqlAlchemy tables defining the project database schema and creates the database if it does not exist.
        """
        if not self.engine.table_names():
            tables = Tables()
            tables.metadata_obj.create_all(self.engine)

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
        dao_name = self.class_mapping[class_.__name__].__name__
        stmt = sqlalchemy.select(self.class_mapping[class_.__name__])
        if query is not None:
            for q in query:
                stmt = stmt.where(sqlalchemy.text(q))
        with Session(self.engine) as session:
            result = session.execute(stmt)
            return [class_({'id': row[dao_name].id, 'name': row[dao_name].name}) for row in result]


class _ShallowSQLite3(_ShallowSQL):
    """
    A concrete shallow SQLite3 persistence inheriting _ShallowSQL and implementing SQLite3-specific engine.
    """

    def __init__(self, dbname):
        super().__init__(dbname)
        self.engine = sqlalchemy.create_engine(f'sqlite+pysqlite:///{self.name}')
        super().create()
