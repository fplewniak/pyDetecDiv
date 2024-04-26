"""
Utility classes to manipulate Image resources
"""

class Dimension:
    """
    A utility class to handle Image resource dimension in ImageResourceDAO ORM objects
    """
    def __init__(self, x, y, z=1, c=1, t=1):
        self.x = x
        self.y = y
        self.c = c
        self.z = z
        self.t = t

    def __composite_values__(self):
        return self.x, self.y, self.z, self.c, self.t

    def __repr__(self):
        return f"Dimension(X={self.x!r}, Y={self.y!r}, Z={self.z!r}, C={self.c!r}, T={self.t!r})"

    def __eq__(self, other):
        return isinstance(other, Dimension) and all([other.x == self.x,
                                                     other.y == self.y,
                                                     other.z == self.z,
                                                     other.c == self.c,
                                                     other.t == self.t,
                                                     ])

    def __ne__(self, other):
        return not self.__eq__(other)

class Shape:
    """
    A utility class to handle Image resource shape in ImageResourceDAO ORM objects
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __composite_values__(self):
        return self.x, self.y

    def __repr__(self):
        return f"Shape(X={self.x!r}, Y={self.y!r}"

    def __eq__(self, other):
        return isinstance(other, Dimension) and all([other.x == self.x,
                                                     other.y == self.y,
                                                     ])

    def __ne__(self, other):
        return not self.__eq__(other)


class Image:
    """
    A utility class to handle 2D Image in ImageResourceDAO ORM objects
    """
    def __init__(self, z=0, c=0, t=0):
        self.c = c
        self.z = z
        self.t = t

    def __composite_values__(self):
        return self.z, self.c, self.t

    def __repr__(self):
        return f"Image(Z={self.z!r}, C={self.c!r}, T={self.t!r})"

    def __eq__(self, other):
        return isinstance(other, Image) and all([other.z == self.z,
                                                     other.c == self.c,
                                                     other.t == self.t,
                                                     ])

    def __ne__(self, other):
        return not self.__eq__(other)
