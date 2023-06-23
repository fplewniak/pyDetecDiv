"""
A set of general utility functions
"""
#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import numpy as np


def singleton(class_):
    """
    Definition of a singleton annotation, creating an object if it does not exist yet or returning the current one if
    it exists

    :param class_: the singleton class
    :return: the singleton instance
    """
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


def round_to_even(value, ceil=True):
    """
    Round a float value to an even integer. If ceil is True then the returned integer is the first even number equal to
    or larger than the rounded integer, otherwise, it is the first smaller or equal even number

    :param value: the value to round to an even number
    :type value: float
    :param ceil: True if the even value should be greater than or equal to the rounded value
    :type ceil: bool
    :return: the rounded value
    :rtype: int
    """
    rounded = int(np.around(value))
    if rounded % 2 != 0:
        if ceil:
            rounded += 1
        else:
            rounded -= 1
    return rounded

def remove_keys_from_dict(dictionary, keys):
    """
    Remove the dictionary entries whose keys are in the key list

    :param dictionary: the dictionary to remove items from
    :type dictionary: dict
    :param keys: the key list
    :type keys: list of str or any object that can be used a dictionary key
    :return: the filtered dictionary
    :rtype: dict
    """
    return dict(filter(lambda item: item[0] not in keys, dictionary.items()))
