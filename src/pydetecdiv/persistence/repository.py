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
    def save_object(self, class_name, record):
        """
        Abstract method enforcing the implementation of a save_object() method in all shallow persistence connectors
        :param class_name: the class name of the object to save
        :type class_name: str
        :param record: the record representing the object
        :type record: dict
        """

    @abc.abstractmethod
    def delete_object(self, class_name, id_):
        """
        Abstract method enforcing the implementation of a delete_object() method in all shallow persistence connectors
        :param class_name: the class name of the object to delete
        :type class_name: str
        :param id_: the id of the object to delete
        :type id_: int
        """

    @abc.abstractmethod
    def get_dataframe(self, class_name, id_list=None):
        """
        Abstract method enforcing the implementation of a method returning a pandas DataFrame containing the list of
        objects of a given class in the current project whose id is in id_list
        :param class_name: the class name of the objects whose list will be returned
        :type class_name: str
        :param id_list: the list of ids of objects to retrieve
        :type id_list: a list of int
        :return: a DataFrame containing the list of objects
        :rtype: DataFrame containing the records representing the requested domain-specific objects
        """

    @abc.abstractmethod
    def get_record(self, class_name, id_):
        """
        Abstract method enforcing the implementation of a method returning an object record of a given
        class whose id is id_
        :param class_name: the class of the requested object
        :type class_name: str
        :param id_: the id of the reaquested object
        :type id_: int
        :return: a record (i.e. dictionary) containing the data for the requested object
        :rtype: dict
        """

    @abc.abstractmethod
    def get_records(self, class_name, id_list):
        """
        Abstract method enforcing the implementation of a method returning the list of all object records of a given
        class specified by its name or a list of objects whose id is in id_list
        :param class_name: the class name of the requested objects
        :type class_name: str
        :param id_list: the list of ids for objects
        :type id_list: list of int
        :return: a list of records (i.e. dictionaries) containing the data for the requested objects
        :rtype: list of dict
        """

    @abc.abstractmethod
    def get_linked_records(self, class_name, parent_class_name, parent_id):
        """
        Abstract method enforcing the implemetnation of a method returning the list of records for all objects of class
         defined by class_name that are linked to object of class parent_class_name with id_ = parent_id
        :param class_name: the class name of the objects to retrieve records for
        :type class_name: str
        :param parent_class_name: the class nae of the parent object
        :type parent_class_name: str
        :param parent_id: the id of the parent object
        :type parent_id: int
        :return: a list of records
        :rtype: list of dict
        """
