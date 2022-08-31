#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from pydetecdiv.domain.dso import DomainSpecificObject


class ROI(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of Regions of interest (ROI)
    """

    @property
    def name(self):
        """
        Returns the name of this ROI
        :return: a str
        """
        return self.data['name']
