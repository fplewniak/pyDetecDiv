import json

import sqlalchemy

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.domain.ROI import ROI


def get_classifications(roi: ROI, run_list: list[int], as_index: bool = True) -> list[int] | list[str]:
    """
    Get the annotations for a ROI as defined in a list of runs

    :param roi: the ROI
    :param run_list: the list of Runs where the ROI was annotated or classified
    :param as_index: bool set to True to return annotations as indices of class_names list, set to False to return
     annotations as class names
    :return: the list of annotated classes by frame
    """
    roi_classes = [-1] * roi.fov.image_resource().image_resource_data().sizeT
    with pydetecdiv_project(PyDetecDiv.project_name) as project:
        results = list(project.repository.session.execute(
                sqlalchemy.text(f"SELECT rc.roi,rc.t,rc.class_name,"
                                f"run.parameters ->> '$.class_names' as class_names, rc.run, run.id_ "
                                f"FROM run, roi_classification as rc "
                                f"WHERE rc.run IN ({','.join([str(i) for i in run_list])}) and rc.roi={roi.id_} "
                                f"AND run.id_=rc.run "
                                f"ORDER BY rc.run ASC;")))
        if results:
            class_names = json.loads(results[0][3])
            if as_index:
                for annotation in results:
                    roi_classes[annotation[1]] = class_names.index(annotation[2])
            else:
                for annotation in results:
                    roi_classes[annotation[1]] = annotation[2]
    return roi_classes


def get_annotation_runs() -> dict:
    """
   Gets previously run prediction runs

    :return: a dictionary containing the ids of all prediction runs corresponding to a given list of classes
    """
    with pydetecdiv_project(PyDetecDiv.project_name) as project:
        results = list(project.repository.session.execute(
                sqlalchemy.text(f"SELECT run.id_,"
                                f"run.parameters ->> '$.annotator' as annotator, "
                                f"run.parameters ->> '$.class_names' as class_names "
                                f"FROM run "
                                f"WHERE (run.command='annotate_rois' OR run.command='import_annotated_rois') "
                                f"ORDER BY run.id_ ASC;")))
        runs = {}
        if results:
            for run in results:
                class_names = json.dumps(json.loads(run[2]))
                if class_names in runs:
                    runs[class_names].append(run[0])
                else:
                    runs[class_names] = [run[0]]
    return runs
