#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
The central class for keeping track of all available objects in a project.
"""
from pandas import DataFrame
from pydetecdiv.persistence.project import open_project
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.domain.ROI import ROI


class Project:
    """
    Project class to keep track of the database connection and providing basic methods to retrieve objects. Actually
    hide repository from other domain classes
    """

    def __init__(self, dbname=None, dbms=None):
        self.repository = open_project(dbname, dbms)
        self.dbname = dbname
        self.new_dso_pool = {}

    def add_new_dso_to_pool(self, dso):
        """
        Adds a newly created domain specific object (has no id) to the pool of objects that need to receive an id and
        to be saved by the persistence layer
        :param dso: the newly created domain-specific object to add to the pool
        :type dso: DomainSpecificObject
        """
        class_name = dso.__class__.__name__
        if class_name not in self.new_dso_pool:
            self.new_dso_pool[class_name] = [dso]
        else:
            self.new_dso_pool[class_name].append(dso)

    def get_objects(self, class_=DomainSpecificObject):
        """
        Get a list of all domain objects of a given class in the current project retrieved from the repository
        :param class_: the class of the domain-specific objects to be returned
        :type class_: class inheriting DomainSpecificObject
        :return: a list of all the objects of that class in the project with all their associated metadata
        :rtype: list of the requested domain-specific objects
        """
        return [class_(project=self, **rec) for rec in self.get_records(class_)]

    def get_dataframe(self, class_=DomainSpecificObject):
        """
        Get a DataFrame containing the list of all domain objects of a given class in the current project
        :param class_: the class of the objects whose list will be returned
        :type class_: class inheriting DomainSpecificObject
        :return: a DataFrame containing the list of objects
        :rtype: DataFrame containing the records representing the requested domain-specific objects
        """
        return DataFrame(self.get_records(class_))

    def get_records(self, class_=DomainSpecificObject) -> list:
        """
        Get a list of records containing all domain objects of a given class in the current project
        :param class_: the class of the objects whose records will be returned
        :type class_: class inheriting DomainSpecificObject
        :return: a list of records representing the requested objects
        :rtype: list of dictionaries (records)
        """
        return self.repository.get_records(class_)

    def get_object_by_id(self, class_=DomainSpecificObject, id_=None) -> DomainSpecificObject:
        """
        Get an object referenced by its id
        :param class_: the class of the requested object
        :param id_: the id reference of the object
        :type class_: class inheriting DomainSpecificObject
        :type id_: int
        :return: the desired object
        :rtype: object (DomainSpecificObject)
        """
        return self.get_objects_by_id(class_, [id_])[0]

    def get_objects_by_id(self, class_=DomainSpecificObject, id_list=None):
        """
        Get a list of objects of one class from a list of id references
        :param class_: the class of the requested objects
        :param id_list: the list of id references
        :type class_: class inheriting DomainSpecificObject
        :type id_list: list of int
        :return: the list of objects
        :rtype: list of objects (DomainSpecificObject)
        """
        if id_list is None:
            obj = self.get_objects(class_)
        else:
            id_list = id_list if isinstance(id_list, list) else [id_list]
            obj = [class_(project=self, **rec) for rec in self.get_records(class_) if rec['id'] in id_list]
        return obj

    def get_roi_list_in_fov(self, fov):
        """
        Get a list of ROIs whose parent is the specified FOV. This method also looks into the pool of new objects to
        fetch newly created ROIs associated with the FOV
        :param fov: FOV object to retrieve the list of associated ROIs
        :type: FOV
        :return: the list of ROIs whose parent is the specified FOV
        :rtype: list of ROI objects
        """
        roi_records = [ROI(project=self, **r) for r in self.repository.get_roi_list_in_fov(fov.id)]
        new_roi = [roi for roi in self.new_dso_pool['ROI'] if roi.fov == fov] if 'ROI' in self.new_dso_pool else []
        return roi_records + new_roi
