#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""
from pydetecdiv.persistence.sqlalchemy.dao.orm import ROI_DAO

class ROI():
    dao_class = ROI_DAO

    def __init__(self, data: ROI_DAO):
        self.data = data

    @property
    def name(self):
        return self.data.name

