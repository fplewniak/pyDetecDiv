#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
The central class for keeping track of all available objects in a project.
"""
from pydetecdiv.persistence.project import open_project
from pydetecdiv.domain.dso import DomainSpecificObject
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

    def save(self, dso):
        """
        Save an object to the repository if it is new or has been modified
        :param dso: the domain-specific object to save
        :type dso: DomainSpecificObject
        :return: the id of the saved object if it was created
        :rtype: int
        """
        id_ = self.repository.save(dso.__class__.__name__, dso.record())
        dso.updated = False
        #print(f'saving {dso}')
        return id_

    def get_object(self, class_=DomainSpecificObject, id_=None) -> DomainSpecificObject:
        """
        Get an object referenced by its id
        :param class_: the class of the requested object
        :param id_: the id reference of the object
        :type class_: class inheriting DomainSpecificObject
        :type id_: int
        :return: the desired object
        :rtype: object (DomainSpecificObject)
        """
        return class_(project=self, **self.repository.get_record(class_.__name__, id_))

    def get_objects(self, class_=DomainSpecificObject, id_list=None):
        """
        Get a list of all domain objects of a given class in the current project retrieved from the repository
        :param class_: the class of the domain-specific objects to be returned
        :type class_: class inheriting DomainSpecificObject
        :param id_list: the list of ids for the objects to retrieve
        :type id_list: list of int
        :return: a list of all the objects of that class in the project with all their associated metadata
        :rtype: list of the requested domain-specific objects
        """
        return [class_(project=self, **rec) for rec in self.repository.get_records(class_.__name__, id_list)]

    def get_roi_list_in_fov(self, fov):
        """
        Get a list of ROIs whose parent is the specified FOV. This method also looks into the pool of new objects to
        fetch newly created ROIs associated with the FOV
        :param fov: FOV object to retrieve the list of associated ROIs
        :type: FOV
        :return: the list of ROIs whose parent is the specified FOV
        :rtype: list of ROI objects
        """
        roi_records = [ROI(project=self, **rec) for rec in self.repository.get_roi_list_in_fov(fov.id_)]
        new_roi = [roi for roi in self.new_dso_pool['ROI'] if roi.fov == fov] if 'ROI' in self.new_dso_pool else []
        return roi_records + new_roi
