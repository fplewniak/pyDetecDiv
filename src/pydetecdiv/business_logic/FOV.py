#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""
from pydetecdiv.database.dao.sqlalchemy.orm import FOV_DAO
from pydetecdiv.business_logic.ROI import ROI


class FOV():
    # A class attribute defining the mapping between the business and data access classes
    dao = FOV_DAO

    def __init__(self, data: FOV_DAO):
        self.data = data['FOV_DAO']

    @property
    def name(self):
        return self.data.name

    @property
    def id(self):
        return self.data.id

    # How should this work ?
    # Ask FOV_DAO to return the list ? (then, who's supposed to keep track of the database engine?)
    # Use a function get_roi_list(self.data) ? (but where to put it, and pass the database engine?)
    # Use a proxy pattern, mediator pattern ? Which pattern is best to use here ?
    #@property
    #def roi_list(self):
    #    return [ROI(roi_dao) for roi_dao in self.data.get_roi_list()]
