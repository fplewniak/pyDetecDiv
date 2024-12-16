#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Custom exceptions to handle errors and enforce requirements
"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydetecdiv.domain.dso import NamedDSO, BoxedDSO


class MissingNameError(Warning):
    """
    Exception raised for missing name in data when creating a NamedDSO
    """

    def __init__(self, obj: 'NamedDSO'):
        self.message = f'Trying to create a {obj.__class__.__name__} (named object) without any name.'
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class JuttingError(Warning):
    """
    Exception raised when a portion of a region (ROI, FOV, image,...) protrudes out of its parent.
    """

    def __init__(self, region: 'BoxedDSO', parent: 'BoxedDSO'):
        r_name = region.__class__.__name__
        p_name = parent.__class__.__name__
        name = region.record['name'] if 'name' in region.record else ''
        self.message = f'{r_name} {name} is protruding out of its parent {p_name} with id {parent.id_}.'
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class OpenProjectError(Exception):
    """
    Exception raised when a project cannot be opened
    """

    def __init__(self, msg: str):
        self.message = f'Cannot open project.\n {msg}'
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ImportImagesError(Exception):
    """
    Exception raised when images cannot be imported
    """

    def __init__(self, msg: str):
        self.message = f'Cannot import images.\n {msg}'
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class UnknownRepositoryTypeError(Exception):
    """
    Exception raised when images cannot be imported
    """

    def __init__(self, msg: str):
        self.message = f'Unknown repository type.\n {msg}'
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
