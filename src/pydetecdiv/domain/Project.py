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
        self.pool = {}

    def save(self, dso):
        """
        Save an object to the repository if it is new or has been modified
        :param dso: the domain-specific object to save
        :type dso: DomainSpecificObject
        :return: the id of the saved object if it was created
        :rtype: int
        """
        id_ = self.repository.save_object(dso.__class__.__name__, dso.record())
        return id_

    def delete(self, dso):
        """
        Delete a domain-specific object
        :param dso: the object to delete
        :type dso: object (DomainSpecificObject)
        """
        self.release_dso(dso.__class__, dso.id_)
        self.repository.delete_object(dso.__class__.__name__, dso.id_)

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
        # return class_(project=self, **self.repository.get_record(class_.__name__, id_))
        return self.build_dso(class_, self.repository.get_record(class_.__name__, id_))

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
        # return [class_(project=self, **rec) for rec in self.repository.get_records(class_.__name__, id_list)]
        return [self.build_dso(class_, rec) for rec in self.repository.get_records(class_.__name__, id_list)]

    def get_roi_list_in_fov(self, fov):
        """
        Get a list of ROIs whose parent is the specified FOV. This method also looks into the pool of new objects to
        fetch newly created ROIs associated with the FOV
        :param fov: FOV object to retrieve the list of associated ROIs
        :type: FOV
        :return: the list of ROIs whose parent is the specified FOV
        :rtype: list of ROI objects
        """
        # roi_records = [ROI(project=self, **rec) for rec in self.repository.get_roi_list_in_fov(fov.id_)]
        roi_records = [self.build_dso(ROI, rec) for rec in self.repository.get_roi_list_in_fov(fov.id_)]
        return roi_records

    def build_dso(self, class_, rec):
        """
        factory method to build a dso of class class_ from record rec or return the pooled object if it was already
        created
        :param class_: the class of the dso to build
        :type class_: type (class)
        :param rec: the record representing the object to build (if id_ is not in the record, the object is a new
        creation
        :type rec: dict
        :return: the requested object
        :rtype: DomainSpecificObject
        """
        if 'id_' in rec and (class_.__name__, rec['id_']) in self.pool:
            return self.pool[(class_.__name__, rec['id_'])]
        obj = class_(project=self, **rec)
        obj.validate()
        self.pool[(class_.__name__, obj.id_)] = obj
        return obj

    def release_dso(self, class_, id_):
        """
        remove a dso from the pool. This should be done when deleting an object with delete() method or to release an
        object that is not needed any more
        :param class_: the class of the object to be removed from the pool
        :type class_: class
        :param id_: the id of the object to be removed from the pool
        :type id_: int
        """
        if (class_.__name__, id_) in self.pool:
            del self.pool[(class_.__name__, id_)]
