from typing import Callable

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.persistence.project import project_exists


def AND(*funcs: Callable[..., bool]) -> Callable[..., bool]:
    return lambda: _and(*funcs)


def OR(*funcs: Callable[..., bool]) -> Callable[..., bool]:
    return lambda: _or(*funcs)


def NOT(*funcs: Callable[..., bool]) -> Callable[..., bool]:
    return lambda: _not(*funcs)


def _and(*funcs):
    results = [func() for func in funcs]
    return all(results)


def _or(*funcs):
    results = [func() for func in funcs]
    return any(results)


def _not(*funcs):
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
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            return project.count_objects('ImageResource') > 0
    return False


def if_data_imageres_is_null() -> bool:
    """
    Enable action if data has no associated image resource
    :param action: the action
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
