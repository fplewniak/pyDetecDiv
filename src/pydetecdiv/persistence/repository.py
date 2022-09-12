#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Definition of the Repository interface accessible from the Business-logic layer and that must be implemented by
concrete repositories.
"""
import abc


class ShallowDb(abc.ABC):
    """
    Abstract class used as an interface to encapsulate project persistence access
    """

    @abc.abstractmethod
    def create(self):
        """
        Abstract method enforcing the implementation in all shallow persistence connectors of a create() method for
        creating the database if it does not exist
        """

    @abc.abstractmethod
    def close(self):
        """
        Abstract method enforcing the implementation of a close() method in all shallow persistence connectors
        """

    @abc.abstractmethod
    def get_dataframe(self, class_, id_list=None):
        """
        Abstract method enforcing the implemetation of a method returning a pandas DataFrame containing the list of
        objects of a given class in the current project whose id is in id_list
        :param class_: the class of the objects whose list will be returned
        :type class_: class inheriting DomainSpecificObject
        :param id_list: the list of ids of objects to retrieve
        :type id_list: a list of int
        :return: a DataFrame containing the list of objects
        :rtype: DataFrame containing the records representing the requested domain-specific objects
        """

    @abc.abstractmethod
    def get_record(self, class_, id_):
        """
        Abstract method enforcing the implementation of a method returning an object record of a given
        class whose id is id_
        :param class_: the class of the requested object
        :type class_: a class
        :param id_list: the id of the reaquested object
        :type id_list: int
        :return: a record (i.e. dictionary) containing the data for the requested object
        :rtype: dict
        """

    @abc.abstractmethod
    def get_records(self, class_, id_list):
        """
        Abstract method enforcing the implementation of a method returning the list of all object records of a given
        class specified by its name or a list of objects whose id is in id_list
        :param class_: the class of the requested objects
        :type class_: a class
        :param id_list: the list of ids for objects
        :type id_list: list of int
        :return: a list of records (i.e. dictionaries) containing the data for the requested objects
        :rtype: list of dict
        """

    @abc.abstractmethod
    def get_roi_list_in_fov(self, fov_id):
        """
        Abstract method enforcing the implementation of a method returning the list of records for all ROI in the FOV
        with id == fov_id
        :param fov_id: the id of the FOV
        :type fov_id: int
        :return: a list of ROIs whose parent if the FOV with id == fov_id
        :rtype: list of dictionaries (records)
        """
