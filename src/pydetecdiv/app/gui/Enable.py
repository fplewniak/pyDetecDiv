from PySide6.QtGui import QAction

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.persistence.project import project_exists


def if_annotations(action: QAction) -> None:
    """
    Enable action if there are ROI annotations in repository
    :param action: the action
    """
    action.setEnabled(False)
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            if (project.count_objects('RoiAnnotations') > 0) and (project.count_objects('Classification') > 0):
                action.setEnabled(True)


def if_class_scheme(action: QAction) -> None:
    """
    Enable action if there is are defined classification schemes in repository
    :param action: the action
    """
    action.setEnabled(False)
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            if project.count_objects('Classification') > 0:
                action.setEnabled(True)


def if_project_exists(action: QAction) -> None:
    """
    Enable action if project exists
    :param action: the action
    """
    action.setEnabled(False)
    if project_exists(PyDetecDiv.project_name):
        action.setEnabled(True)


def if_rois(action: QAction) -> None:
    """
    Enable action if there is are ROIs in repository
    :param action: the action
    """
    action.setEnabled(False)
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            if project.count_objects('ROI') > 0:
                action.setEnabled(True)


def if_data_imageres_is_null(action: QAction) -> None:
    """
    Enable action if data has no associated image resource
    :param action: the action
    """
    action.setEnabled(False)
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            for data in project.get_objects('Data'):
                if project.count_links('ImageResource', data) == 0:
                    action.setEnabled(True)
                    break


def if_missing_image_resources(action: QAction) -> None:
    """
    Enable action if there is are undefined image resources in repository
    :param action: the action
    """
    action.setEnabled(False)
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            if project.count_orphan_data_files():
                action.setEnabled(True)
