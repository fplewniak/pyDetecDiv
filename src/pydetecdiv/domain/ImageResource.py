#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
from pydetecdiv.domain.dso import DomainSpecificObject
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.domain.Dataset import Dataset


class ImageResource(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of Image resources
    """

    def __init__(self, dataset, fov=None, xdim=1024, ydim=1024, zdim=1, cdim=1, tdim=1,
                 xyscale=1, tscale=1, zscale=1, xyunit=1e-6, zunit=1e-6, tunit=60, **kwargs):
        super().__init__(**kwargs)
        self._dataset = dataset.id_ if isinstance(dataset, Dataset) else dataset
        self._fov = fov.id_ if isinstance(fov, FOV) else fov
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.cdim = cdim
        self.tdim = tdim
        self.xyscale = xyscale
        self.xyunit = xyunit
        self.zscale = zscale
        self.zunit = zunit
        self.tscale = tscale
        self.tunit = tunit
        self.validate(updated=False)

    def record(self, no_id=False):
        """
        Returns a record dictionary of the current Image resource

        :param no_id: if True, does not return id_ (useful for transferring from one project to another)
        :type no_id: bool
        :return: record dictionary
        :rtype: dict
        """
        record = {
            'dataset': self._dataset,
            'xdim': self.xdim,
            'ydim': self.ydim,
            'zdim': self.zdim,
            'cdim': self.cdim,
            'tdim': self.tdim,
            'xyscale': self.xyscale,
            'xyunit': self.xyunit,
            'zscale': self.zscale,
            'zunit': self.zunit,
            'tscale': self.tscale,
            'tunit': self.tunit,
            'fov': self._fov,
            'uuid': self.uuid
        }
        if not no_id:
            record['id_'] = self.id_
        return record
