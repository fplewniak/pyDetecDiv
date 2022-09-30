#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to images
"""
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.domain.ImageData import ImageData


class Image(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of 2D images
    """

    def __init__(self, image_data=None, locator=None, mimetype=None, drift=(0, 0), c=0, z=0, t=0, **kwargs):
        super().__init__(**kwargs)
        self._image_data = image_data.id_ if isinstance(image_data, ImageData) else image_data
        self.locator = locator
        self.mimetype = mimetype
        self.drift = drift
        self.c_ = c
        self.z_ = z
        self.t = t
        self.validate(updated=False)

    def check_validity(self):
        """
        Checks the current Image validity
        """
        ...

    @property
    def image_data(self):
        """
        property returning the image data this image is related to
        :return: the parent ImageData object
        :rtype: ImageData
        """
        return self.project.get_object('ImageData', self._image_data)

    @image_data.setter
    def image_data(self, image_data):
        self._image_data = image_data.id_ if isinstance(image_data, ImageData) else image_data
        self.validate()

    @property
    def roi(self):
        """
        property returning the ROI this image is related to
        :return: the parent ROI object
        :rtype: ROI
        """
        return self.project.get_linked_objects('ROI', to=self)[0]

    @property
    def fov(self):
        """
        property returning the ROI this image is related to
        :return: the parent ROI object
        :rtype: ROI
        """
        return self.project.get_linked_objects('FOV', to=self)[0]

    @property
    def top_left(self):
        """
        The top left coordinates of the image relative to the file. These are retrieved from the parent ImageData
        object
        :return: top left coordinates
        :rtype: tuple of int
        """
        return self.roi.top_left

    @property
    def bottom_right(self):
        """
        The bottom right coordinates of the image relative to the file. These are retrieved from the parent ImageData
        object
        :return: the coordinates of the bottom-right corner
        :rtype: a tuple of two int
        """
        return self.roi.bottom_right

    @property
    def c(self):
        """
        The channel of this image
        :return: the channel index
        :rtype: int
        """
        return self.c_

    @c.setter
    def c(self, c):
        self.c_ = c
        self.validate()

    @property
    def z(self):
        """
        The z-index of this image
        :return: the z index
        :rtype: int
        """
        return self.z_

    @z.setter
    def z(self, z):
        self.z_ = z
        self.validate()

    @property
    def t(self):
        """
        The time index of this image
        :return: the time index
        :rtype: int
        """
        return self.t_

    @t.setter
    def t(self, t):
        self.t_ = t
        self.validate()

    def record(self, no_id=False):
        """
        Returns a record dictionary of the current ROI
        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to
        another
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'image_data': self._image_data,
            'locator': self.locator,
            'mimetype': self.mimetype,
            'drift': self.drift,
            'c': self.c,
            'z': self.z,
            't': self.t,
        }
        if not no_id:
            record['id_'] = self.id_
        return record
