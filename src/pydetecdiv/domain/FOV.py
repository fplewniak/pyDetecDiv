#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
from pydetecdiv.domain.dso import NamedDSO, BoxedDSO


class FOV(NamedDSO, BoxedDSO):
    """
    A business-logic class defining valid operations and attributes of Fields of view (FOV)
    """

    def __init__(self, comments=None, **kwargs):
        super().__init__(**kwargs)
        self._comments = comments
        self.validate(updated=False)

    def delete(self):
        """
        Delete the current FOV, first deleting all linked ROIs that would be consequently left orphaned
        """
        for roi in self.roi_list:
            roi.delete()
        self.project.delete(self)

    def check_validity(self):
        """
        Checks the current FOV is valid
        """
        ...

    def record(self, no_id=False):
        """
        Returns a record dictionary of the current FOV
        :param no_id: if True, does not return id_ (useful for transferring from one project to another)
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'name': self.name,
            'comments': self.comments,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
            'size': self.size
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    def __repr__(self):
        return f'{self.record()}'

    @property
    def comments(self):
        """
        comments property of FOV
        :return: the comments
        :rtype: str
        """
        return self._comments

    @comments.setter
    def comments(self, comments):
        self._comments = comments
        self.validate()

    @property
    def roi_list(self):
        """
        Returns the list of ROI objects whose parent if the current FOV
        :return: the list of associated ROIs
        :rtype: list of ROI objects
        """
        return self.project.get_linked_objects('ROI', to=self)

    @property
    def image_data(self):
        """
        Returns the list of ROI objects whose parent if the current FOV
        :return: the list of associated ROIs
        :rtype: list of ROI objects
        """
        return self.project.get_linked_objects('ImageData', to=self)

    def add_image_data(self, image_data):
        """
        Link an ImageData object to the current FOV
        :param image_data: ImageData object
        :type image_data: ImageData
        """
        self.project.link_objects(self, image_data)

    def detach_image_data(self, image_data):
        """
        Remove the image data from the list of ImageData objects linked to the current FOV
        :param image_data: ImageData object
        :type image_data: ImageData
        """
        self.project.unlink_objects(self, image_data)
