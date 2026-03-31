from pydetecdiv.domain.Classification import Classification
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.Run import Run
from pydetecdiv.domain.dso import DomainSpecificObject


class RoiAnnotations(DomainSpecificObject):
    """
    A business-logic class defining valid operations and attributes of Image resources
    """

    def __init__(self, roi, t, annotation, classification, run):
        self._roi = roi.id_ if isinstance(roi, ROI) else roi
        self._classification = classification.id_ if isinstance(classification, Classification) else classification
        self._annotation = self.classification.classes.index(annotation) if isinstance(annotation, str) else annotation
        self._run = run.id_ if isinstance(run, Run) else run


    @property
    def classification(self):
        return self.project.get_object('Classifier', self._classification)

    @property
    def roi(self):
        return self.project.get_object('ROI', self._roi)

    @property
    def annotation(self):
        return self.project.get_object('RoiAnnotation', self._annotation)

    @property
    def class_name(self):
        return self.classification.classes[self._annotation]

    @property
    def run(self):
        return self.project.get_object('Run', self._run)
