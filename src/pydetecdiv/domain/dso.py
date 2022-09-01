#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Domain specific generic attributes and methods shared by all domain objects. Other domain classes (except Project)
should inherit from this class
"""
from pydetecdiv.domain.Project import Project


class DomainSpecificObject:
    """
    A business-logic class defining valid operations and attributes common to all domain-specific classes
    """

    def __init__(self, project: Project, data: dict = None):
        """
        Initialization of the object with a link to the Project it belongs to and associated data as a dictionary
        :param project: the project object this object belongs to
        :param data: a dictionary containing the associated data. This dictionary should at least contain one entry with
        the 'id' of this object in the project.
        """
        self.data = data if data is not None else {'id': None}
        self.project = project

    @property
    def id(self):
        """
        Returns the id of this object
        :return: an integer
        """
        return self.data['id']
