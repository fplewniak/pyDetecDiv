#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
from pydetecdiv.domain.dso import *


class FOV(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of Fields of view (FOV)
    """

    # How should this work ?
    # Ask FOV_DAO to return the list ? (then, who's supposed to keep track of the persistence engine?)
    # Use a function get_roi_list(self) ? could be in repository, for SQL dbms, sqlalchemy could use Tables for that,
    # determining the appropriate table from the self class name and sending the appropriate SQL queries accordingly
    # Use a proxy pattern, mediator pattern ? Which pattern is best to use here ?
    # @property
    # def roi_list(self):
    #    roi = self.project.get_dataframe(ROI)
    #    return roi[roi.fov == self.id]
