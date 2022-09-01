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
    def get_object_list(self, class_name: str = None, as_list=False):
        """
        Abstract method enforcing the implementation of a method returning the list of all objects of a given class
        specified by its name
        :param class_name: the class name of the requested objects
        :param dataframe: if True, returns a DataFrame else returns a list of dictionaries
        :return: a DataFrame or a list of dictionaries containing the data for the requested objects
        """
