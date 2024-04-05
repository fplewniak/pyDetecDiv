#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Project persistence management for persistence layer
"""
import glob
import os
import shutil

from pydetecdiv.exceptions import UnknownRepositoryTypeError
from pydetecdiv.utils.path import stem
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
    match dbms:
        case 'SQLite3':
            dbname = dbname if dbname is not None else get_config_value('project.sqlite', 'database')
            workspace = get_config_value('project', 'workspace')
            db = ShallowSQLite3(f'{workspace}/{dbname}.db')
        case _:
            raise NotImplementedError(f'{dbms} is not implemented')
    return db


def list_projects(dbms: str = None):
    """
    Return a list of projects corresponding to a given database manager system

    :param dbms: the dbms
    :type dbms: str
    :return: the list of projects
    :rtype: list of str
    """
    dbms = get_config_value('project', 'dbms') if dbms is None else dbms
    project_list = []
    match dbms:
        case 'SQLite3':
            workspace = get_config_value('project', 'workspace')
            project_list = [stem(db_file) for db_file in glob.glob(f'{workspace}/*.db')]
        case _:
            raise UnknownRepositoryTypeError(f'{dbms} is not implemented')
    return project_list

def delete_project(dbname: str = None, dbms: str = None):
    """
    Deletes the project from its name

    :param dbms: the dbms
    :type dbms: str
    """
    dbms = get_config_value('project', 'dbms') if dbms is None else dbms
    project_list = list_projects()
    match dbms:
        case 'SQLite3':
            workspace = get_config_value('project', 'workspace')
            if dbname in project_list:
                os.remove(os.path.join(workspace, f'{dbname}.db'))
                shutil.rmtree(os.path.join(workspace, dbname))
        case _:
            raise UnknownRepositoryTypeError(f'{dbms} is not implemented')
