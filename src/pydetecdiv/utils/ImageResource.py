"""
Utility classes to manipulate Image resources
"""
from __future__ import annotations

class Dimension:
    """
    A utility class to handle Image resource dimension in ImageResourceDAO ORM objects
    """

    def __init__(self, x: int, y: int, z: int = 1, c: int = 1, t: int = 1):
        self.x = x
        self.y = y
        self.c = c
        self.z = z
        self.t = t

    def __composite_values__(self) -> tuple[int, int, int, int, int]:
        return self.x, self.y, self.z, self.c, self.t

    def __repr__(self) -> str:
        return f"Dimension(X={self.x!r}, Y={self.y!r}, Z={self.z!r}, C={self.c!r}, T={self.t!r})"

    def __eq__(self, other: Dimension) -> bool:
        return isinstance(other, Dimension) and all([other.x == self.x,
                                                     other.y == self.y,
                                                     other.z == self.z,
                                                     other.c == self.c,
                                                     other.t == self.t,
                                                     ])

    def __ne__(self, other: Dimension) -> bool:
        return not self.__eq__(other)


class Shape:
    """
    A utility class to handle Image resource shape in ImageResourceDAO ORM objects
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __composite_values__(self) -> tuple[int, int]:
        return self.x, self.y

    def __repr__(self) -> str:
        return f"Shape(X={self.x!r}, Y={self.y!r}"

    def __eq__(self, other: Shape) -> bool:
        return isinstance(other, Shape) and all([other.x == self.x,
                                                     other.y == self.y,
                                                     ])

    def __ne__(self, other: Shape) -> bool:
        return not self.__eq__(other)


class Image:
    """
    A utility class to handle 2D Image in ImageResourceDAO ORM objects
    """

    def __init__(self, z: int=0, c: int=0, t: int=0):
        self.c = c
        self.z = z
        self.t = t

    def __composite_values__(self) -> tuple[int, int, int]:
        return self.z, self.c, self.t

    def __repr__(self) -> str:
        return f"Image(Z={self.z!r}, C={self.c!r}, T={self.t!r})"

    def __eq__(self, other: Image) -> bool:
        return isinstance(other, Image) and all([other.z == self.z,
                                                 other.c == self.c,
                                                 other.t == self.t,
                                                 ])

    def __ne__(self, other: Image) -> bool:
        return not self.__eq__(other)
