#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
The central class for keeping track of all available objects in a project.
"""
from pandas import DataFrame
from pydetecdiv.persistence.project import open_project


class Project:
    """
    Project class to keep track of the database connection and providing basic methods to retrieve objects. Actually
    hide repository from other domain classes
    """

    def __init__(self, dbname: str = None, dbms: str = None):
        self.repository = open_project(dbname, dbms)
        self.dbname = dbname

    def get_objects(self, class_: type) -> list:
        """
        Get a list of all domain objects of a given class in the current project retrieved from the repository
        :param class_: the class of the objects to be returned
        :return: a list of all the objects of that class in the project with all their associated metadata
        """
        return [class_(self, o) for o in self.repository.get_object_list(class_.__name__)]

    def get_dataframe(self, class_: type) -> DataFrame:
        """
        Get a DataFrame containing the list of all domain objects of a given class in the current project
        :param class_: the class of the objects whose list will be returned
        :return: a DataFrame containing the list of objects
        """
        return self.repository.get_object_list(class_.__name__, as_list=False)

    def get_object_by_id(self, class_: type, id_: int) -> object:
        """
        Get an object referenced by its id
        :param class_: the class of the requested object
        :param id_: the id reference of the object
        :return: the desired object
        """
        return self.get_objects_by_id(class_, [id_])[0]

    def get_objects_by_id(self, class_: type, id_list: list = None) -> list:
        """
        Get a list of objects of one class from a list of id references
        :param class_: the class of the requested objects
        :param id_list: the list of id references
        :return: the list of objects
        """
        if id_list is None:
            obj = self.get_objects(class_)
        else:
            id_list = id_list if isinstance(id_list, list) else [id_list]
            obj_df = self.get_dataframe(class_)
            obj_df_selection = [obj_df[obj_df.id == id_] for id_ in id_list]
            obj = [class_(self, o.to_dict(orient='records')[0]) for o in obj_df_selection]
        return obj
