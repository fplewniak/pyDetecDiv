#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Project persistence management for persistence layer
"""
from pydetecdiv.settings import get_config_value
from pydetecdiv.persistence.repository import ShallowDb
from pydetecdiv.persistence.sqlalchemy.repositories import ShallowSQLite3


def open_project(dbname: str = None, dbms: str = None) -> ShallowDb:
    """
    A function to open a shallow persistence from its name. The default type of persistence is defined in the [project]
    sections of the configuration file settings.ini
    :param dbname: the persistence name
    :param dbms: A str specifying the database management system to use with the project
    :return: a shallowDb abstract connector encapsulating the concrete connectors
    """
    dbms = get_config_value('project', 'dbms') if dbms is None else dbms
    if dbms == 'SQLite3':
        dbname = dbname if dbname is not None else get_config_value('project.sqlite', 'database')
        db = ShallowSQLite3(dbname)
    else:
        raise NotImplementedError(f'{dbms} is not implemented')
    return db
