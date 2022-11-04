#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to images
"""
from collections import namedtuple
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.domain.ImageData import ImageData


class Image(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of 2D images
    """

    def __init__(self, image_data=None, resource=None, location=(0, 0, 0), mimetype=None, drift=(0, 0), layer=0,
                 frame=0, order='zct', **kwargs):
        super().__init__(**kwargs)
        self._image_data = image_data.id_ if isinstance(image_data, ImageData) else image_data
        self._resource = resource
        self._mimetype = mimetype
        self._drift = drift
        self._order = order
        self._location = dict(zip(self.order, location))
        self._layer = layer
        self._frame = frame
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
    def resource(self):
        """
        Returns the image resource (file, etc.) storing the image
        :return: the image resource
        :rtype: str
        """
        return self._resource

    @resource.setter
    def resource(self, resource):
        self._resource = resource
        self.validate()

    @property
    def mimetype(self):
        """
        The mime-type of the image resource
        :return: the mime-type
        :rtype: str
        """
        return self._mimetype

    @mimetype.setter
    def mimetype(self, mimetype):
        self._mimetype = mimetype
        self.validate()

    @property
    def location(self):
        """
        The location of the image in the resource as indexes for z, c, and t
        :return: the location of the image
        :rtype: a named tuple
        """
        Location = namedtuple('Location', list(self.order))
        return Location(**self._location)

    @location.setter
    def location(self, location):
        self._location = dict(zip(self.order, location))
        self.validate()

    @property
    def order(self):
        """
        The order the z, c and t dimensions are stored in the resource
        :return: the order of z, c and t dimensions
        :rtype: str
        """
        return self._order

    @order.setter
    def order(self, order='zct'):
        self._order = order
        self.validate()

    @property
    def c(self):
        """
        The channel of this image
        :return: the channel index
        :rtype: int
        """
        return self._location['c']

    @c.setter
    def c(self, c):
        self._location['c'] = c
        self.validate()

    @property
    def z(self):
        """
        The z-index of this image in the image resource defined by self._resource
        :return: the z index
        :rtype: int
        """
        return self._location['z']

    @z.setter
    def z(self, z):
        self._location['z'] = z
        self.validate()

    @property
    def layer(self):
        """
        The actual z position (layer) of this image
        :return: the actual layer index
        :rtype: int
        """
        return self._layer

    @layer.setter
    def layer(self, layer):
        self._layer = layer
        self.validate()

    @property
    def t(self):
        """
        The time index of this image in the image resource defined by self._resource
        :return: the time index
        :rtype: int
        """
        return self._location['t']

    @t.setter
    def t(self, t):
        self._location['t'] = t
        self.validate()

    @property
    def frame(self):
        """
        The actual time frame index of this image equal to the number of frame_interval since the first frame
        :return: the actual frame index
        :rtype: int
        """
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = frame
        self.validate()

    @property
    def drift(self):
        """
        The (x, y) drift of this image relative to the first one
        :return: position drift
        :rtype: tuple of int
        """
        return self._drift

    @property
    def time(self):
        """
        The actual time of this image computed from the image actual index and frame_interval
        :return:
        """
        return self.frame * self.image_data.frame_interval

    @drift.setter
    def drift(self, drift=(0,0)):
        self._drift = drift
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
            'layer': self._layer,
            'frame': self._frame,
            'drift': self._drift,
            'resource': self._resource,
            'location': self._location,
            'order': self._order,
            'mimetype': self._mimetype,
        }
        if not no_id:
            record['id_'] = self.id_
        return record
