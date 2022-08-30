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
        Abstract method enforcing the implementation of a create() method in all shallow persistence connectors
        """

    @abc.abstractmethod
    def close(self):
        """
        Abstract method enforcing the implementation of a close() method in all shallow persistence connectors
        """

    @abc.abstractmethod
    def get_objects(self, class_: type = None, query=None):
        """
        Abstract method enforcing the implementation of method returning a list of objects corresponding to a query
        :param class_: the class of the requested objects
        :param query: a query represented by a dictionary specifying the constraints that must be met by object
        attributes
        """



