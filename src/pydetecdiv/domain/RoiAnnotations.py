from typing import Any

from pydetecdiv.domain.Classification import Classification
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.Run import Run
from pydetecdiv.domain.dso import DomainSpecificObject


class RoiAnnotations(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of ROI annotations
    """

    def __init__(self, roi, t, annotation, classification, run, key_val, **kwargs):
        super().__init__(**kwargs)
        self._roi = roi.id_ if isinstance(roi, ROI) else roi
        self._classification = classification.id_ if isinstance(classification, Classification) else classification
        self._annotation = self.classification.classes.index(annotation) + 1 if isinstance(annotation, str) else annotation
        self.t = t
        self._run = run.id_ if isinstance(run, Run) else run
        self.key_val = key_val
        self.validate(updated=False)

    @property
    def classification(self):
        return self.project.get_object('Classifier', self._classification)

    @property
    def roi(self):
        return self.project.get_object('ROI', self._roi)

    @property
    def annotation(self):
        return self._annotation

    @property
    def class_name(self):
        return self.classification.classes[self._annotation - 1]

    @property
    def run(self):
        return self.project.get_object('Run', self._run)

    def record(self, no_id: bool = False) -> dict[str, Any]:
        """
        Returns a record dictionary of the current ROI

        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to another
        :type no_id: bool
        :return: record dictionary
        """
        record = {
            'roi'           : self._roi,
            't'             : self.t,
            'annotation'    : self._annotation,
            'classification': self._classification,
            'run'           : self._run,
            'uuid'          : self.uuid,
            'key_val'       : self.key_val,
            }
        if not no_id:
            record['id_'] = self.id_
        return record
