#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
The central class for keeping track of all available objects in a project.
"""
from pandas import DataFrame, Series
from pydetecdiv.persistence.project import open_project


class Project:
    """
    Project class to keep track of the database connection and providing basic methods to retrieve objects. Actually
    hide repository from other domain classes
    """
    def __init__(self, dbname: str = None, dbms: str = None):
        self.repository = open_project(dbname, dbms)
        self.dbname = dbname

    def get_objects(self, class_: type = None) -> list:
        """
        Get a list of all domain objects of a given class in the current project retrieved from the repository
        :param class_: the class of the objects to be returned
        :return: a list of all the objects of that class in the project with all their associated metadata
        """
        return [class_(self, o) for o in self.repository.get_objects(class_.__name__)]

    def get_dataframe(self, class_: type = None) -> DataFrame:
        """
        Get a DataFrame containing the list of all domain objects of a given class in the current project
        :param class_: the class of the objects whose list will be returned
        :return: a DataFrame containing the list of objects
        """
        return DataFrame([Series(o) for o in self.repository.get_objects(class_.__name__)])
