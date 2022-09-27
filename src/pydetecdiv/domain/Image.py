#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to images
"""
# from pydetecdiv.exceptions import JuttingError
from pydetecdiv.domain.dso import BoxedDSO, DsoWithImageData
from pydetecdiv.domain.ImageData import ImageData
from pydetecdiv.domain.FileResource import FileResource


class Image(BoxedDSO, DsoWithImageData):
    """
    A business-logic class defining valid operations and attributes of images
    """

    def __init__(self, image_data=None, resource=None, drift=(0, 0), z=0, t=0, **kwargs):
        super().__init__(**kwargs)
        self._image_data = (image_data if isinstance(image_data, ImageData)
                                          or image_data is None else self.project.get_object('ImageData', image_data))
        self._resource = (resource if isinstance(resource, FileResource)
                                      or resource is None else self.project.get_object('FileResource', resource))
        self.drift = drift
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
        return self._image_data

    @image_data.setter
    def image_data(self, image_data):
        self._image_data = (
            image_data if isinstance(image_data, ImageData) or image_data is None else self.project.get_object(
                'ImageData', image_data))
        self.validate()

    @property
    def resource(self):
        """
        property returning the image data this image is related to
        :return: the parent ImageData object
        :rtype: ImageData
        """
        return self._resource

    @resource.setter
    def resource(self, resource):
        self._resource = (
            resource if isinstance(resource, FileResource) or resource is None else self.project.get_object(
                'FileResource', resource))
        self.validate()

    @property
    def top_left(self):
        """
        The top left coordinates of the image relative to the file. These are retrieved from the parent ImageData
        object
        :return: top left coordinates
        :rtype: tuple of int
        """
        return self.image_data.top_left

    @top_left.setter
    def top_left(self, top_left):
        self.image_data.top_left = top_left

    @property
    def bottom_right(self):
        """
        The bottom right coordinates of the image relative to the file. These are retrieved from the parent ImageData
        object
        :return: the coordinates of the bottom-right corner
        :rtype: a tuple of two int
        """
        return self.image_data.bottom_right

    @bottom_right.setter
    def bottom_right(self, bottom_right):
        self.image_data.bottom_right = bottom_right

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
            'image_data': self._image_data.id_,
            'resource': self._resource.id_,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
            'drift': self.drift,
            'z': self.z,
            't': self.t,
            'size': self.size
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    def __repr__(self):
        return f'{self.record()}'
