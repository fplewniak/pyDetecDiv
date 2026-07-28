"""
A set of general utility functions
"""
#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Any
import numpy as np
import polars
from ndtiff import NDTiffDataset

from pydetecdiv.utils.path import files_in_dir


def singleton(class_) -> Callable:
    """
    Definition of a singleton annotation, creating an object if it does not exist yet or returning the current one if
    it exists

    :param class_: the singleton class
    :return: the singleton instance
    """
    instances = {}

    def getinstance(*args, **kwargs) -> Any:
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


class Singleton:
    """
    A class defining a Singleton (can only be instantiated once in an application)
    """
    instance = None
    initialized = False

    def __new__(cls, *args, **kwargs) -> Any:
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance

    def __init__(self, *args, **kwargs):
        if self.initialized:
            return
        self.initialized = True


class BidirectionalIterator:
    """
    An iterator that can go backwards
    """

    def __init__(self, data: list[Any]):
        self.data = data
        self.index = -1  # Start before the first element

    def __iter__(self) -> BidirectionalIterator:
        return self

    def __next__(self) -> Any:
        if self.index < len(self.data) - 1:
            self.index += 1
            return self.data[self.index]
        raise StopIteration

    def __previous__(self) -> Any:
        if self.index > 0:
            self.index -= 1
            return self.data[self.index]
        raise StopIteration("No previous element")


def previous(iterator: BidirectionalIterator) -> Any:
    """
    the previous element

    :param iterator: the iterator
    :return: the previous element
    """
    return iterator.__previous__()


def round_to_even(value: float, ceil: bool = True) -> int:
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


def remove_keys_from_dict(dictionary: dict[str | Any, Any], keys: list[str | Any]):
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


def split_list(arr: list[Any] | np.ndarray[Any], sep: list[Any] | Any, max_length: int = None,
               constant_length: bool = True) -> list:
    """
    Split a list

    :param arr: the array to split
    :param sep: the separator
    :param max_length: the maximum length of the split
    :param constant_length: whether all sublists should be of the same length
    :return: the split list
    """
    arr = np.array(arr)
    sep = sep if isinstance(sep, list) else [sep]
    indices = np.where(np.diff([arr == s for s in sep]))[0] + 1
    sublists = []
    start = 0
    for idx in indices:
        num_sections = int((idx - start) / max_length + 0.5) if max_length else 1
        sublists.extend(np.array_split(arr[start:idx], num_sections))
        start = idx

    if start < len(arr):
        num_sections = int((len(arr) - start) / max_length + 0.5) if max_length else 1
        sublists.extend(np.array_split(arr[start:], num_sections))

    if constant_length:
        return [sublist for sublist in sublists if len(sublist) == max_length]
    return sublists


def flatten_list(list_of_lists: list[list[Any]]) -> list[Any]:
    """
    Flatten a list of lists

    :param list_of_lists: the list of lists to flatten
    :return: the flat list
    """
    return [x for sublist in list_of_lists for x in sublist]


def increment_string(s: str) -> str:
    """
    Increments a string by one: a becomes b, which in turn becomes c, etc. After z comes aa, and so on.

    :param s: the string to increment
    :return: the incremented string
    """
    s = list(s)
    i = len(s) - 1
    while i >= 0:
        if s[i] != 'z':
            s[i] = chr(ord(s[i]) + 1)  # bump this char
            return ''.join(s)
        s[i] = 'a'  # reset and carry over
        i -= 1
    return 'a' + ''.join(s)  # expand if overflow (zzz → aaaa)


def check_is_ndtiff(directory: str | bytes) -> bool:
    """
    Check whether the specified path is a NDTiff path and enables the Ok button if it is
    """
    return directory != '' and os.path.isfile(os.path.join(str(directory), 'NDTiff.index'))


def check_contains_tiff(directory: str) -> bool:
    """
    Check the directory contains tiff files
    :param directory: the directory name
    """
    return len(files_in_dir(directory, ['*.tiff', '*.tif'])) > 0


def count_ndtiff(dirpath: str) -> int:
    """
    Count NDTiff datasets
    :param dirpath: the path to the datasets
    """
    ndtiff_dirs = [f for f in glob.glob(dirpath) if os.path.isdir(f) and check_is_ndtiff(f)]
    ndtiff_dir_count = 0
    for ndtiff_dir in ndtiff_dirs:
        ndtiff_ds = NDTiffDataset(str(ndtiff_dir))
        df = polars.DataFrame(ndtiff_ds.get_image_coordinates_list())
        dims_df = df.group_by(by='position').agg(polars.col('time').max(), polars.col('z').max(), polars.col('channel').max())
        ndtiff_dir_count += dims_df.select(polars.len()).item()

    print(f'counting ndtiff: {ndtiff_dir_count} NDTIff FOV dataset')
    return ndtiff_dir_count


def count_image_dir(dirpath) -> int:
    """
    Count image files in directories
    :param dirpath: the path to the directories
    """
    image_dirs = [f for f in glob.glob(dirpath) if os.path.isdir(f) and check_contains_tiff(f)]
    file_count = 0
    for image_dir in image_dirs:
        # file_count += len(glob.glob(image_dir + '/*.tiff')) + len(glob.glob(image_dir + '/*.tif'))
        file_count += len(files_in_dir(str(image_dir), ['*.tiff', '*.tif']))
    print(f'counting image files: {file_count} image files')
    return file_count


def count_metadata(filepath) -> int:
    """
    Count image files using MicroManager metadata files
    :param filepath: the path to the metadata file(s)
    """
    metadata_file_names = [f for f in glob.glob(filepath) if os.path.isfile(f)]
    file_count = 0
    for metadata_file_name in metadata_file_names:
        with open(metadata_file_name) as metadata_file:
            metadata = json.load(metadata_file)
            file_count += len([v for k, v in metadata.items() if k.startswith('Metadata-')])
    print(f'counting metadata: {file_count} image files')
    return file_count
