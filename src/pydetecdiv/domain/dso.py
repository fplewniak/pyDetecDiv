#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Domain specific generic attributes and methods shared by domain objects. Other domain classes (except Project)
should inherit from these classes according to their type.
"""
import uuid
from pydetecdiv.exceptions import MissingNameError
from pydetecdiv.utils.Shapes import Box


class DomainSpecificObject:
    """
    A business-logic class defining valid operations and attributes common to all domain-specific classes
    """

    def __init__(self, project=None, **kwargs):
        """
        Initialization of the object with a link to the Project it belongs to and associated _data as a dictionary

        :param project: the project object this object belongs to
        :param **kwargs: a dictionary containing the record used to create the domain-specific object.
        :type project: Project object
        :type **kwargs: dict
        """
        self.project = project
        if 'id_' not in kwargs:
            self.id_ = None
        else:
            self.id_ = kwargs['id_']

        if 'uuid' not in kwargs:
            self.uuid = str(uuid.uuid4())
        else:
            self.uuid = kwargs['uuid']
        self._data = kwargs

    def __eq__(self, o):
        """
        Defines equality of domain-specific objects as having the same id and same class

        :param o: the other dso to compare with the current one
        :type o: DomainSpecificObject
        :return: True if both objects are the same
        :rtype: bool
        """
        is_eq = [self.id_ == o.id_, self.__class__ == o.__class__]
        return all(is_eq)

    def __repr__(self):
        return f'{self.record()}'

    def delete(self):
        """
        Delete the current object
        """
        self.project.delete(self)

    def check_validity(self):
        """
        Checks the validity of the current object
        """

    def validate(self, updated=True):
        """
        Validate the current object and pass newly created and updated object to project for saving modifications. Sets
        the id of the object for new objects.
        This method should be called at the creation or at each modification of an object (i.e. in the __init__ and all
        setter methods)
        """
        self.check_validity()
        if updated or self.id_ is None:
            self.id_ = self.project.save(self)

    def record(self, no_id=False):
        """
        Returns a record dictionary of the current DSO

        :return: record dictionary
        :rtype: dict
        """
        exclude_keys = set(['id_']) if no_id else set()
        return {k: self._data[k] for k in set(list(self._data.keys())) - set(exclude_keys)}


class NamedDSO(DomainSpecificObject):
    """
    A domain-specific class for objects with a name.
    """

    def __init__(self, name=None, **kwargs):
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

    @name.setter
    def name(self, name):
        """
        Returns the name of this object

        :return: the name of the current object
        :rtype: str
        """
        self._name = name
        self.validate()


class BoxedDSO(DomainSpecificObject):
    """
    A domain-specific class for objects that can be represented as a rectangular box (i.e. having top left and bottom
    right corners).
    """

    def __init__(self, top_left=None, bottom_right=None, **kwargs):
        super().__init__(**kwargs)
        self._top_left = top_left
        self._bottom_right = bottom_right

    @property
    def box(self):
        """
        Returns a Box object that can represent the current object

        :return: a box with the same coordinates
        :rtype: Box
        """
        return Box(self.top_left, self.bottom_right)

    @property
    def x(self):
        return self.box.top_left[0]

    @property
    def y(self):
        return self.box.top_left[1]

    @property
    def width(self):
        return self.box.width

    @property
    def height(self):
        return self.box.height

    @property
    def top_left(self):
        """
        The top-left corner of the Box in the coordinate system

        :return: the coordinates of the top-left corner
        :rtype: a tuple of two int
        """
        return self._top_left

    @top_left.setter
    def top_left(self, top_left):
        self._top_left = top_left
        self.validate()

    @property
    def bottom_right(self):
        """
        The bottom-right corner of the Box in the coordinate system

        :return: the coordinates of the bottom-right corner
        :rtype: a tuple of two int
        """
        return self._bottom_right

    @bottom_right.setter
    def bottom_right(self, bottom_right):
        self._bottom_right = bottom_right
        self.validate()

    @property
    def size(self):
        """
        The size (dimension) of the object obtained from its associated box

        :return: the dimension of the boxed object
        :rtype: a tuple of two int
        """
        return self.box.size
