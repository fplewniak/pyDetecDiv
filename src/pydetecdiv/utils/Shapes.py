#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
A utility module defining classes of shapes
"""
from __future__ import annotations


class Box:
    """
    A utility class defining a Box for use by all classes representing two-dimensional objects such as images.
    """

    def __init__(self, top_left: tuple[int, int] = None, bottom_right: tuple[int, int] = None):
        """
        Create a box from the coordinates of top left and bottom right corners

        :param top_left: top left corner of the box
        :param bottom_right: bottom right corner of the box
        :type top_left: tuple of two int
        :type bottom_right: tuple of two int
        """
        self.top_left = top_left
        self.bottom_right = bottom_right

    @property
    def x(self) -> int:
        return self._top_left[0]

    @property
    def y(self) -> int:
        return self._top_left[1]

    @property
    def top_left(self) -> tuple[int, int]:
        """
        top left corner of the box

        :return: coordinates of the top left corner of the box
        :rtype: tuple of two int
        """
        return self._top_left

    @top_left.setter
    def top_left(self, top_left: tuple[int, int]) -> None:
        if top_left is None:
            self._top_left = (0, 0)
        else:
            self._top_left = top_left

    @property
    def bottom_right(self) -> tuple[int, int]:
        """
        bottom right corner of the box

        :return: coordinates of the bottom right corner of the box
        :rtype: tuple of two int
        """
        return self._bottom_right

    @bottom_right.setter
    def bottom_right(self, bottom_right: tuple[int, int]):
        self._bottom_right = bottom_right

    @property
    def width(self) -> int:
        """
        width of the box

        :return: width of the box
        :rtype: int
        """
        return self.bottom_right[0] - self.top_left[0] + 1

    @property
    def height(self) -> int:
        """
        height of the box

        :return: height of the box
        :rtype: int
        """
        return self.bottom_right[1] - self.top_left[1] + 1

    @property
    def size(self) -> tuple[int, int]:
        """
        size dimensions of the box, i.e. width and height

        :return: size dimensions of the box
        :rtype: tuple of two int
        """
        return self.width, self.height

    def lies_in(self, other: Box) -> bool:
        """
        Checks whether the current object lies within the boundaries of another box

        :param other: the other box
        :type other: Box
        :return: True if the current box is totally comprised in the other one
        :rtype: bool
        """
        return (self.top_left[0] >= other.top_left[0]
                and self.bottom_right[0] <= other.bottom_right[0]
                and self.top_left[1] >= other.top_left[1]
                and self.bottom_right[1] <= other.bottom_right[1])
