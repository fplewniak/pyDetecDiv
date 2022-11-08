#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
The central class for keeping track of all available objects in a project.
"""
import itertools
from collections import defaultdict
from pydetecdiv.persistence.project import open_project
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.ImageData import ImageData
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.domain.Image import Image


class Project:
    """
    Project class to keep track of the database connection and providing basic methods to retrieve objects. Actually
    hides repository from other domain classes. This class handles actual records and domain-specific objects while the
    repository deals with records and data-access objects. Data exchange between Repository and Project objects is
    achieved through the use of dict representing standardized records.
    """
    classes = {
        'ROI': ROI,
        'FOV': FOV,
        'ImageData': ImageData,
        'Image': Image,
    }

    def __init__(self, dbname=None, dbms=None):
        self.repository = open_project(dbname, dbms)
        self.dbname = dbname
        self.pool = defaultdict(DomainSpecificObject)

    def import_images(self, source_path):
        """
        Import images from a source path. All files corresponding to the path will be imported.
        :param source_path: the source path (glob pattern)
        :type source_path: str
        """
        self.repository.import_images(source_path)

    def create_dsos_from_raw_data(self, source, keys_, regex):
        """
        Create domain-specific objects from raw data using a regular expression applied to a bioimageit database field
        or a combination thereof specified by source. DSOs to create are specified by the values in keys.
        :param source: the database field or combination of fields to apply the regular expression to
        :type source: str or callable returning a str
        :param keys_: the list of classes created objects belong to
        :type keys_: tuple of str
        :param regex: regular expression defining the DSOs' names
        :type regex: regular expression str
        """
        # TODO add a file argument for metadata to complete information about dsos, for example position of a ROI, etc.
        # TODO ultimately creation of dsos should be the responsibility of DSO classes themselves in order to keep this
        # TODO class as small as possible
        df = self.repository.annotate_raw_data(source, keys_, regex)
        if 'FOV' in keys_:
            fov_names = [f.name for f in self.get_objects('FOV')]
            for fov in df.FOV.drop_duplicates().values:
                if fov not in fov_names:
                    FOV(project=self, name=fov, top_left=(0, 0), bottom_right=(999, 999))
                #TODO link this fov to image data uuid

        for k in keys_:
            print(df.filter([k, 'uuid']))

    def save_record(self, class_name, record):
        """
        Creates and saves an object of class named class_name from its record without requiring the creation of a DSO.
        This method can be useful for creating associated objects with mutual dependency.
        :param class_name: the class name of the object to create
        :type class_name: str
        :param record: the record representing the object to create
        :type record: dict
        :return: the id of the created object
        :rtype: int
        """
        id_ = self.repository.save_object(class_name, record)
        return id_

    def save(self, dso):
        """
        Save an object to the repository if it is new or has been modified
        :param dso: the domain-specific object to save
        :type dso: DomainSpecificObject
        :return: the id of the saved object if it was created
        :rtype: int
        """
        id_ = self.repository.save_object(dso.__class__.__name__, dso.record())
        if (dso.__class__.__name__, id_) not in self.pool:
            self.pool[(dso.__class__.__name__, id_)] = dso
        return id_

    def delete(self, dso):
        """
        Delete a domain-specific object
        :param dso: the object to delete
        :type dso: object (DomainSpecificObject)
        """
        del self.pool[dso.__class__.__name__, dso.id_]
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
        :param id_list: the list of ids for the objects to be retrieved
        :type id_list: list of int
        :return: a list of all the objects of that class in the project with all their associated metadata
        :rtype: list of the requested domain-specific objects
        """
        if class_name == 'ROI':
            return self._get_rois(id_list)
        return [self.build_dso(class_name, rec) for rec in self.repository.get_records(class_name, id_list)]

    def _get_rois(self, id_list=None):
        """
        Gets ROIs using FOV.roi_list properties for all FOVs in order to show the initial ROIs only for FOVs that have
        no other defined ROI. This method is used by the generic get_objects method to deal with the special case of
        initial ROIs
        :param id_list: the list of ids for the ROIs to be retrieved
        :type id_list: list of int
        :return: a list of the requested ROIs
        :rtype: list of ROI objects
        """
        all_rois = itertools.chain(*[fov.roi_list for fov in self.get_objects('FOV')])
        if id_list is None:
            return list(all_rois)
        return [roi for roi in all_rois if roi.id_ in id_list]

    def has_links(self, class_name, to=None):
        """
        Checks whether there are links to a given object from objects of a given class.
        :param class_name: the name of the class from which the existence of links is tested
        :type class_name: str
        :param to: the object which is tested for existence of links from class_name class
        :type to: DomainSpecificObject object
        :return: True if links exist, False otherwise
        :rtype: bool
        """
        if self.repository.get_linked_records(class_name, to.__class__.__name__, to.id_):
            return True
        return False

    def count_links(self, class_name, to=None):
        """
        Counts the number of objects of a given class having a link to an object
        :param class_name: the name of the class whose links to the specified object are counted
        :type class_name: str
        :param to: the object to which links are counted
        :type to: DomainSpecificObject object
        :return: the number of objects of class class_name that are linked to the specified object
        :rtype: bool
        """
        return len(self.repository.get_linked_records(class_name, to.__class__.__name__, to.id_))

    def get_linked_objects(self, class_name, to=None):
        """
        A method returning the list of all objects of class defined by class_name that are linked to an object specified
        by argument to=
        Note that for ROIs linked to a FOV, this method also returns the initial ROI (full-FOV) even if other ROIs have
        been created. If initial ROIs are not desired, then the FOV.roi_list property should be used instead.
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

    def link_objects(self, dso1: DomainSpecificObject, dso2: DomainSpecificObject):
        """
        Create a direct link between two objects. This method only works for objects that have a direct logical
        connection defined in an association table. It does not work to create transitive links with intermediate
        objects
        :param dso1: first domain-specific object to link
        :type dso1: object
        :param dso2: second domain-specific object to link
        :type dso2: object
        """
        self.repository.link(dso1.__class__.__name__, dso1.id_, dso2.__class__.__name__, dso2.id_, )

    def unlink_objects(self, dso1: DomainSpecificObject, dso2: DomainSpecificObject):
        """
        Delete a direct link between two objects. This method only works for objects that have a direct logical
        connection defined in an association table. It does not work to delete transitive links with intermediate
        objects
        :param dso1: first domain-specific object to unlink
        :type dso1: object
        :param dso2: second domain-specific object to unlink
        :type dso2: object
        """
        self.repository.unlink(dso1.__class__.__name__, dso1.id_, dso2.__class__.__name__, dso2.id_, )

    def build_dso(self, class_name, rec):
        """
        factory method to build a dso of class class_ from record rec or return the pooled object if it was already
        created. Note that if the object was already in the pool, values in the record are not used to update the
        object. To make any change to the object's attribute, the object's class methods and setter should be used
        instead as they ensure that values are checked for validity and new attributes are saved to the persistence
        back-end
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
