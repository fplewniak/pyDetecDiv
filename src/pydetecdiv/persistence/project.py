#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Project persistence management for persistence layer
"""
import glob
import json
import os
import xml

import yaml

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
    dbms = get_config_value('project', 'dbms') if dbms is None else dbms
    project_list = []
    match dbms:
        case 'SQLite3':
            workspace = get_config_value('project', 'workspace')
            project_list = [stem(db_file) for db_file in glob.glob(f'{workspace}/*.db')]
        case _:
            raise NotImplementedError(f'{dbms} is not implemented')
    return project_list


def list_tools():
    """
    Provide a list of available tools arranged by categories
    :return: the list of available tools and categories
    :rtype: dict
    """
    toolbox_path = get_config_value('paths', 'toolbox')
    json_data = json.load(open(os.path.join(toolbox_path, 'toolboxes.json')))
    tool_list = {c['name']: [] for c in json_data['categories']}
    for current_path, subs, files in os.walk(os.path.abspath(os.path.join(toolbox_path, 'tools'))):
        for file in files:
            if file.endswith('.xml'):
                tree = xml.etree.ElementTree.parse(os.path.join(current_path, file))
                # print(tree.getroot().attrib)
                # print(tree.getroot().find('command').text)
                shed_file = os.path.join(current_path, '.shed.yml')
                with open(shed_file) as file:
                    shed_file_content = yaml.load(file, Loader=yaml.FullLoader)
                    for category in shed_file_content["categories"]:
                        tool_list[category].append([tree.getroot().get('name'), tree.getroot().get('version')])
    return tool_list
