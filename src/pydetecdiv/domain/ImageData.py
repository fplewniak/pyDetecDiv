#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from pydetecdiv.domain.dso import BoxedDSO
from pydetecdiv.domain.FileResource import FileResource


class ImageData(BoxedDSO):
    """
    A business-logic class defining valid operations and attributes of Regions of interest (ROI)
    """

    def __init__(self, file_resource=None, name=None, channel=0, stacks=1, frames=1, interval=None, orderdims='xyzct',
                 path=None, mimetype=None, **kwargs):
        super().__init__(**kwargs)
        self._file_resource = (file_resource if isinstance(file_resource, FileResource) or file_resource is None
                               else self.project.get_object(FileResource, file_resource))
        self.name = name
        self.channel = channel
        self.stacks = stacks
        self.frames = frames
        self.interval = interval
        self.orderdims = orderdims
        self.path = path
        self.mimetype = mimetype
        self.validate(updated=False)

    def check_validity(self):
        """
        Checks the current ImageData object is valid
        """
        ...

    @property
    def file_resource(self):
        """
        property returning the File resource object where this ImageData is stored
        :return: the parent FileResource object
        :rtype: FileResource
        """
        return self._file_resource

    @file_resource.setter
    def file_resource(self, file_resource):
        self._file_resource = (file_resource if isinstance(file_resource, FileResource) or file_resource is None
                               else self.project.get_object(FileResource, file_resource))
        self.validate()

    @property
    def fov_list(self):
        """
        Returns the list of ROI objects whose parent if the current FOV
        :return: the list of associated ROIs
        :rtype: list of ROI objects
        """
        return self.project.get_linked_objects('FOV', self)

    @property
    def bottom_right(self):
        """
        The bottom-right corner of the ROI in the FOV
        :return: the coordinates of the bottom-right corner
        :rtype: a tuple of two int
        """
        return self._bottom_right
        # return (self.fov.size[0] - 1 if self._bottom_right[0] == -1 else self._bottom_right[0],
        #         self.fov.size[1] - 1 if self._bottom_right[1] == -1 else self._bottom_right[1])

    @bottom_right.setter
    def bottom_right(self, bottom_right):
        self._bottom_right = bottom_right
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
            'name': self.name,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
            'channel': self.channel,
            'stacks': self.stacks,
            'frames': self.frames,
            'interval': self.interval,
            'orderdims': self.orderdims,
            'file_resource': self.file_resource.id_,
            'path': self.path,
            'mimetype': self.mimetype,
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    def __repr__(self):
        return f'{self.record()}'
