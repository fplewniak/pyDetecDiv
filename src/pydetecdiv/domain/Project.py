#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
The central class for keeping track of all available objects in a project.
"""
from pydetecdiv.persistence.project import open_project
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.ImageData import ImageData
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.domain.FileResource import FileResource


class Project:
    """
    Project class to keep track of the database connection and providing basic methods to retrieve objects. Actually
    hide repository from other domain classes
    """
    classes = {
        'ROI': ROI,
        'FOV': FOV,
        'ImageData': ImageData,
        'FileResource': FileResource,
    }

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
        print(dso.record)
        id_ = self.repository.save_object(dso.__class__.__name__, dso.record())
        return id_

    def delete(self, dso):
        """
        Delete a domain-specific object
        :param dso: the object to delete
        :type dso: object (DomainSpecificObject)
        """
        self.release_dso(dso.__class__.__name__, dso.id_)
        self.repository.delete_object(dso.__class__.__name__, dso.id_)

    def get_object(self, class_name, id_=None) -> DomainSpecificObject:
        """
        Get an object referenced by its id
        :param class_name: the class of the requested object
        :param id_: the id reference of the object
        :type class_name: class inheriting DomainSpecificObject
        :type id_: int
        :return: the desired object
        :rtype: object (DomainSpecificObject)
        """
        return self.build_dso(class_name, self.repository.get_record(class_name, id_))

    def get_objects(self, class_name, id_list=None):
        """
        Get a list of all domain objects of a given class in the current project retrieved from the repository
        :param class_name: the class name of the domain-specific objects to be returned
        :type class_name: str
        :param id_list: the list of ids for the objects to retrieve
        :type id_list: list of int
        :return: a list of all the objects of that class in the project with all their associated metadata
        :rtype: list of the requested domain-specific objects
        """
        return [self.build_dso(class_name, rec) for rec in self.repository.get_records(class_name, id_list)]

    def get_linked_objects(self, class_name, to=None):
        """
        A method returning the list of all objects of class defined by class_name that are linked to an object specified
        by linked_to
        :param class_name: the class name of the objects to retrieve
        :type class_name: str
        :param to: the object the retrieve objects should be linked to
        :type to: DomainSpecificObject
        :return: the list of objects linked to linked_to object
        :rtype: list of objects
        """
        object_list = [self.build_dso(class_name, rec) for rec in
                       self.repository.get_linked_records(class_name, to.__class__.__name__, to.id_)]
        return object_list

    def link_objects(self, dso1, dso2):
        """
        Create a direct link between two objects. This method only works for objects that have a direct logical
        connection. It does not work to create transitive links with intermediate objects.
        :param dso1: first domain-specific object to link
        :type dso1: object
        :param dso2: second domain-specific object to link
        :type dso2: object
        """
        self.repository.link(dso1.__class__.__name__, dso1.id_, dso2.__class__.__name__, dso2.id_, )

    def unlink_objects(self, dso1, dso2):
        """
        Delete a direct link between two objects. This method only works for objects that have a direct logical
        connection. It does not work to delete transitive links with intermediate objects.
        :param dso1: first domain-specific object to unlink
        :type dso1: object
        :param dso2: second domain-specific object to unlink
        :type dso2: object
        """
        self.repository.unlink(dso1.__class__.__name__, dso1.id_, dso2.__class__.__name__, dso2.id_, )

    def build_dso(self, class_name, rec):
        """
        factory method to build a dso of class class_ from record rec or return the pooled object if it was already
        created
        :param class_name: the class name of the dso to build
        :type class_name: str
        :param rec: the record representing the object to build (if id_ is not in the record, the object is a new
        creation
        :type rec: dict
        :return: the requested object
        :rtype: DomainSpecificObject
        """
        if 'id_' in rec and (class_name, rec['id_']) in self.pool:
            return self.pool[(class_name, rec['id_'])]
        obj = Project.classes[class_name](project=self, **rec)
        self.pool[(class_name, obj.id_)] = obj
        return obj

    def release_dso(self, class_name, id_):
        """
        remove a dso from the pool. This should be done when deleting an object with delete() method or to release an
        object that is not needed any more
        :param class_name: the class name of the object to be removed from the pool
        :type class_name: str
        :param id_: the id of the object to be removed from the pool
        :type id_: int
        """
        if (class_name, id_) in self.pool:
            del self.pool[(class_name, id_)]
