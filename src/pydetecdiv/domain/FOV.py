#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to Fields Of View
"""


class FOV():
    """
    A business-logic class defining valid operations and attributes of Fields of view (FOV)
    """

    def __init__(self, data: dict):
        self.data = data

    @property
    def name(self):
        """
        Returns the name of this FOV
        :return: a str
        """
        return self.data['name']

    @property
    def id(self):
        """
        Returns the id of this FOV
        :return: an integer
        """
        return self.data['id']

    # How should this work ?
    # Ask FOV_DAO to return the list ? (then, who's supposed to keep track of the persistence engine?)
    # Use a function get_roi_list(self.data) ? (but where to put it, and pass the persistence engine?)
    # Use a proxy pattern, mediator pattern ? Which pattern is best to use here ?
    # @property
    # def roi_list(self):
    #    return [ROI(roi_dao) for roi_dao in self.data.get_roi_list()]
