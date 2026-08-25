"""
Functions for the determination of enable status of actions
"""
import glob
import os
from typing import Callable

import tables

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.persistence.project import project_exists


def AND(*funcs: Callable[..., bool]) -> Callable[..., bool]:
    """
    Defines a combination of functions with the AND operator
    :param funcs: the functions to combine
    """
    return lambda: _and(*funcs)


def OR(*funcs: Callable[..., bool]) -> Callable[..., bool]:
    """
    Defines a combination of functions with the OR operator
    :param funcs: the functions to combine
    """
    return lambda: _or(*funcs)


def NOT(*funcs: Callable[..., bool]) -> Callable[..., bool]:
    """
    Defines a combination of functions with the NOT operator
    :param funcs: the functions to combine
    """
    return lambda: _not(*funcs)


def _and(*funcs: Callable[..., bool]) -> bool:
    """
    Return True if all functions return True
    :param funcs: the functions to test
    """
    results = [func() for func in funcs]
    return all(results)


def _or(*funcs: Callable[..., bool]) -> bool:
    """
    Return True if at least one of the functions returns True
    :param funcs: the functions to test
    """
    results = [func() for func in funcs]
    return any(results)


def _not(*funcs: Callable[..., bool]) -> bool:
    """
    Return True if all functions return False
    :param funcs: the functions to test
    """
    return not _and(*funcs)


def if_annotations() -> bool:
    """
    Enable action if there are ROI annotations in repository
    :param action: the action
    """
    if project_exists(PyDetecDiv.project_name):
        return _and(if_annotated_rois, if_class_scheme)
    return False


def if_annotated_rois() -> bool:
    """
    Return True if there are annotated ROIs
    """
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            return project.count_objects('RoiAnnotations') > 0
    return False


def if_class_scheme() -> bool:
    """
    Enable action if there is are defined classification schemes in repository
    :param action: the action
    """
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            return project.count_objects('Classification') > 0
    return False


def if_project_exists() -> bool:
    """
    Enable action if project exists
    :param action: the action
    """
    return project_exists(PyDetecDiv.project_name)


def if_rois() -> bool:
    """
    Enable action if there is are ROIs in repository
    :param action: the action
    """
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            return project.count_objects('ROI') > 0
    return False


def if_image_resources() -> bool:
    """
    Return True if there are image resources
    """
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            return project.count_objects('ImageResource') > 0
    return False


def if_data_imageres_is_null() -> bool:
    """
    Return True if data has no associated image resource
    """
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            for data in project.get_objects('Data'):
                if project.count_links('ImageResource', data) == 0:
                    return True
    return False


def if_missing_image_resources() -> bool:
    """
    Enable action if there is are undefined image resources in repository
    :param action: the action
    """
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            return project.count_orphan_data_files() > 0
    return False


def if_drift_correction() -> bool:
    """
    Return True if the project contains drift correction files
    """
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            return any((ir.drift is not None) for ir in project.get_objects('ImageResource'))
    return False


def if_exists_roi_hdf5() -> bool:
    """
    Return True if the project contains a HDF5 ROI file
    """
    if project_exists(PyDetecDiv.project_name):
        file_paths = glob.glob(os.path.join(PyDetecDiv.tools['cnrs.plewniak.roiseqhdf5creator'].working_dir, '*.h5'))
        return bool(file_paths)
    return False

def if_hdf5_has_targets() -> bool:
    if project_exists(PyDetecDiv.project_name):
        file_paths = glob.glob(os.path.join(PyDetecDiv.tools['cnrs.plewniak.roiseqhdf5creator'].working_dir, '*.h5'))
        return any([tables.open_file(f, mode='r').__contains__('/targets') for f in file_paths])
    return False
