#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Custom exceptions to handle errors and enforce requirements
"""


class MissingNameError(Exception):
    """
    Exception raised for missing name in data when creating a NamedDSO
    """

    def __init__(self, obj: object):
        self.message = f'Trying to create a {obj.__class__.__name__} (named object) without any name.'
        super().__init__(self.message)

    def __str__(self):
        return self.message
