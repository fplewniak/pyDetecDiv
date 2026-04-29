from PySide6.QtGui import QAction

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.persistence.project import project_exists


def if_annotations(action: QAction):
    action.setEnabled(False)
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            if (project.count_objects('RoiAnnotations') > 0) and (project.count_objects('Classification') > 0):
                action.setEnabled(True)

def if_class_scheme(action: QAction):
    action.setEnabled(False)
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            if project.count_objects('Classification') > 0:
                action.setEnabled(True)


def if_project_exists(action: QAction):
    action.setEnabled(False)
    if project_exists(PyDetecDiv.project_name):
        action.setEnabled(True)


def if_rois(action: QAction):
    action.setEnabled(False)
    if project_exists(PyDetecDiv.project_name):
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            if project.count_objects('ROI') > 0:
                action.setEnabled(True)
