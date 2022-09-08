#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Domain specific generic attributes and methods shared by all domain objects. Other domain classes (except Project)
should inherit from this class
"""
from pydetecdiv.exceptions import MissingNameError

class DomainSpecificObject:
    """
    A business-logic class defining valid operations and attributes common to all domain-specific classes
    """

    def __init__(self, project=None, **kwargs):
        """
        Initialization of the object with a link to the Project it belongs to and associated data as a dictionary
        :param project: the project object this object belongs to
        :param **kwargs: a dictionary containing the record used to create the domain-specific object.
        :type project: Project object
        :type **kwargs: dict
        """
        self.project = project
        if 'id' not in kwargs:
            self.id = None
        else:
            self.id = kwargs['id']
        # if self.id is None:
        #     self.project.add_new_dso_to_pool(self)
        self.data = kwargs

    def check_validity(self):
        """
        Checks the validity of the current object
        """
        pass

    def validate(self):
        """
        Validate the current object and pass newly created object pool in current project if it has no id
        This method should be called
        """
        self.check_validity()
        if self.id is None:
            self.project.add_new_dso_to_pool(self)


class NamedDSO(DomainSpecificObject):
    """
    A domain-specific class for objects with a name.
    """

    def __init__(self, name=None,  **kwargs):
        if name is None:
            raise MissingNameError(self)
        super().__init__(**kwargs)
        self._name = name

    @property
    def name(self):
        """
        Returns the name of this object
        :return: the name of the current object
        :rtype: str
        """
        return self._name


class ImageAssociatedDSO(DomainSpecificObject):
    """
    A domain-specific class for objects associated with an image, and therefore having a shape.
    """

    def __init__(self, shape=(None, None), **kwargs):
        super().__init__(**kwargs)
        self._shape = shape

    @property
    def shape(self):
        """
        Shape property of object
        :return: x and y dimensions of the associated image
        :rtype: tuple of two int
        """
        return self._shape

    @shape.setter
    def shape(self, shape=(None, None)):
        self._shape = shape
