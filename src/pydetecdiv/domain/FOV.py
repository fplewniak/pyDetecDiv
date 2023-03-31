#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
from pydetecdiv.domain.dso import NamedDSO, BoxedDSO, DsoWithImageData


class FOV(NamedDSO, BoxedDSO, DsoWithImageData):
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
            'size': self.size,
            'uuid': self.uuid
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    @property
    def info(self):
        """
        Returns ready-to-print information about FOV
        :return: FOV information
        :rtype: str
        """
        return f"""
Name:                 {self.name}
Size:                 {self.size}
number of ROI:        {len(self.roi_list)}
number of datasets:   {len(self.project.get_linked_objects('Dataset', to=self))}
number of data files: {len(self.project.get_linked_objects('Data', to=self))}
Comments:             {self.comments}
        """

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
    def data(self):
        """
        Property returning the Data objects associated to this FOV
        :return: the data associated to this FOV
        :rtype: list of Data objects
        """
        data = self.project.get_linked_objects('Data', to=self)
        return data

    @property
    def roi_list(self):
        """
        Returns the list of ROI objects whose parent if the current FOV
        :return: the list of associated ROIs
        :rtype: list of ROI objects
        """
        return self.project.get_linked_objects('ROI', to=self)
        # roi_list = self.project.get_linked_objects('ROI', to=self)
        # return roi_list if len(roi_list) == 1 else roi_list[1:]

    @property
    def image_list(self):
        """
        Return a list of images related to this FOV
        :return: list of images
        :rtype: list of Image objects
        """
        return self.project.get_linked_objects('Image', to=self)

    @property
    def initial_roi(self):
        """
        Return the initial ROI, created at the creation of this FOV and representing the full FOV
        :return: the initial ROI
        :rtype: ROI object
        """
        return self.project.get_linked_objects('ROI', to=self)[0]

    def image_resource(self, dataset='data'):
        dataset_uuid = self.project.get_named_object('Dataset', dataset).uuid
        data_list = [data for data in self.data if data.dataset_ == dataset_uuid]
        print(data_list)

