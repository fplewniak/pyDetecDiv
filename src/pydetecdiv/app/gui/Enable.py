from typing import Callable

from PySide6.QtGui import QAction

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.persistence.project import project_exists


def if_annotations() -> bool:
    """
    Enable action if there are ROI annotations in repository
    :param action: the action
    """
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            return (project.count_objects('RoiAnnotations') > 0) and (project.count_objects('Classification') > 0)
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


def _and(func1: Callable[..., bool], func2: Callable[..., bool]) -> bool:
    return func1() and func2()


def _or(func1: Callable[..., bool], func2: Callable[..., bool]) -> bool:
    return func1() or func2()


def _and_not(func1: Callable[..., bool], func2: Callable[..., bool]) -> bool:
    return func1() and not func2()
