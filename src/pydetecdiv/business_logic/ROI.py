#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Regions Of Interest
"""

class ROI():

    def __init__(self, data: dict):
        self.data = data

    @property
    def name(self):
        return self.data['name']

    @property
    def id(self):
        return self.data['id']

