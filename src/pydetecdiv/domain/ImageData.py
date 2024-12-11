#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from collections import namedtuple
from pandas import DataFrame
from pydetecdiv.domain.dso import NamedDSO
from pydetecdiv.domain import ROI


class ImageData(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of 5D Image data related to ROIs
    """

    def __init__(self, roi=None, shape=(1000, 1000, 1, 1, 1), stack_interval=None, frame_interval=None,
                 orderdims='xyzct', **kwargs):
        super().__init__(**kwargs)
        self._roi = roi.id_ if isinstance(roi, ROI) else roi
        self.orderdims = orderdims
        self._shape = dict(zip(self.orderdims, shape))
        self.stack_interval = stack_interval
        self.frame_interval = frame_interval
        self.validate(updated=False)

    def check_validity(self):
        """
        Checks the current ImageData object is valid and consistent with its image content. If
        not, the shape values are updated
        """
        if (self.id_ is not None) and len(self.image_list):
            shape = {'x': self._shape['x'],
                     'y': self._shape['y'],
                     'z': len(set(i.layer for i in self.image_list)),
                     'c': len(set(i.c for i in self.image_list)),
                     't': len(set(i.frame for i in self.image_list))
                     }
            if shape != self._shape:
                self._shape = shape
                self.id_ = self.project.save(self)

    @property
    def shape(self):
        """
        The shape of the ImageData 5D matrix
        :return: the shape of the matrix
        :rtype: tuple of int
        """
        self.check_validity()
        Shape = namedtuple('Shape', list(self.orderdims))
        return Shape(**self._shape)

    @shape.setter
    def shape(self, shape=(1000, 1000, 1, 1, 1)):
        self._shape = dict(zip(self.orderdims, shape))
        self.validate()

    @property
    def roi(self):
        """
        property returning the ROI object corresponding to the ImageData
        :return: the parent ROI object
        :rtype: ROI
        """
        return self.project.get_object('ROI', self._roi)

    @roi.setter
    def roi(self, roi):
        self._roi = roi.id_ if isinstance(roi, ROI) else roi
        self.validate()

    @property
    def fov(self):
        """
        Returns the list of FOV objects associated to the current Image data
        :return: the list of associated FOV
        :rtype: list of FOV objects
        """
        return self.project.get_linked_objects('FOV', to=self)[0]

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
            videos = df.sort_values(by=['layer', 'frame']).groupby('layer')
            videos_rec = [videos.get_group(z).to_dict(orient='records') for z in videos.groups]
            return [[self.project.get_object('Image', f['id_']) for f in v] for v in videos_rec]
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
            stacks = df.sort_values(by=['frame', 'layer']).groupby('frame')
            stacks_rec = [stacks.get_group(frame).to_dict(orient='records') for frame in stacks.groups]
            return [[self.project.get_object('Image', z_rec['id_']) for z_rec in s] for s in stacks_rec]
        return []

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
            'roi': self._roi,
            'shape': self._shape,
            'stack_interval': self.stack_interval,
            'frame_interval': self.frame_interval,
            'orderdims': self.orderdims,
            'uuid': self.uuid
        }
        if not no_id:
            record['id_'] = self.id_
        return record
