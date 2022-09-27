#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from pandas import DataFrame
from pydetecdiv.domain.dso import BoxedDSO
from pydetecdiv.domain.FileResource import FileResource
from pydetecdiv.domain.ROI import ROI


class ImageData(BoxedDSO):
    """
    A business-logic class defining valid operations and attributes of Regions of interest (ROI)
    """

    def __init__(self, file_resource=None, roi=None, name=None, channel=0, stack_interval=None,
                 frame_interval=None, orderdims='xyzct', path=None, mimetype=None, **kwargs):
        super().__init__(**kwargs)
        self._file_resource = (file_resource if isinstance(file_resource, FileResource) or file_resource is None
                               else self.project.get_object('FileResource', file_resource))
        self._roi = (roi if isinstance(roi, ROI) or roi is None
                               else self.project.get_object('ROI', roi))
        self.name = name
        self.channel = channel
        self.stack_interval = stack_interval
        self.frame_interval = frame_interval
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
                               else self.project.get_object('FileResource', file_resource))
        self.validate()

    @property
    def roi(self):
        """
        property returning the File resource object where this ImageData is stored
        :return: the parent FileResource object
        :rtype: FileResource
        """
        return self._roi

    @roi.setter
    def roi(self, roi):
        self._roi = (roi if isinstance(roi, ROI) or roi is None
                               else self.project.get_object('ROI', roi))
        self.validate()

    @property
    def fov_list(self):
        """
        Returns the list of FOV objects associated to the current Image data
        :return: the list of associated FOV
        :rtype: list of FOV objects
        """
        return self.project.get_linked_objects('FOV', to=self)

    @property
    def roi_list(self):
        """
        Returns the list of ROI objects associated to the current Image data
        :return: the list of associated ROIs
        :rtype: list of ROI objects
        """
        return self.project.get_linked_objects('ROI', to=self)

    @property
    def image_list(self):
        """
        Returns the list of ROI objects associated to the current Image data
        :return: the list of associated ROIs
        :rtype: list of ROI objects
        """
        return self.project.get_linked_objects('Image', to=self)

    @property
    def videos(self):
        """
        Returns a list of lists of associated images grouped by z-index. Each sublist can be considered as a video.
        :return: the list of video
        :rtype: a list of list of Image objects
        """
        if len(self.image_list):
            df = DataFrame.from_records([img.record() for img in self.image_list])
            videos = df.sort_values(by=['z', 't']).groupby('z')
            videos_rec = [videos.get_group(z).to_dict(orient='records') for z in videos.groups]
            return [[self.project.get_object('Image', frame_rec['id_']) for frame_rec in video_rec] for video_rec in
                    videos_rec]
        return []


    @property
    def stacks(self):
        """
        Returns a list of lists of associated images grouped by time index. Each sublist can be considered as a stacked
        image.
        :return: the list of stacks
        :rtype: a list of list of Image objects
        """
        if len(self.image_list):
            df = DataFrame.from_records([img.record() for img in self.image_list])
            stacks = df.sort_values(by=['t', 'z']).groupby('t')
            stacks_rec = [stacks.get_group(frame).to_dict(orient='records') for frame in stacks.groups]
            return [[self.project.get_object('Image', z_rec['id_']) for z_rec in stack_rec] for stack_rec in stacks_rec]
        return []

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
            'roi': self._roi.id_,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
            'channel': self.channel,
            'stack_interval': self.stack_interval,
            'frame_interval': self.frame_interval,
            'orderdims': self.orderdims,
            'resource': self.file_resource.id_,
            'path': self.path,
            'mimetype': self.mimetype,
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    def __repr__(self):
        return f'{self.record()}'
