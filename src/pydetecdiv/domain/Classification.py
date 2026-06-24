from typing import Any

from pydetecdiv.domain.dso import NamedDSO


class Classification(NamedDSO):
    """
    A business-logic class defining valid operations and attributes of Classification schemas
    """

    def __init__(self, classes:list['str'] | None = None, key_val: dict[str, Any] | None = None,**kwargs):
        super().__init__(**kwargs)
        self.classes = classes if classes is not None else []
        self.key_val = key_val if key_val is not None else {}
        self.validate(updated=False)

    def delete(self) -> None:
        """
        Deletes this ROI if and only if it is not the full-FOV one which should serve to keep track of original data.
        """
        self.project.delete(self)

    def runs(self):
        return self.project.get_linked_objects('Run', self)

    def record(self, no_id: bool = False) -> dict[str, Any]:
        """
        Returns a record dictionary of the current Classification schemas

        :param no_id: if True, the id_ is not passed included in the record to allow transfer from one project to another
        :type no_id: bool
        :return: record dictionary
        """
        record = {
            'name': self.name,
            'classes': self.classes,
            'uuid': self.uuid,
            'key_val': self.key_val,
        }
        if not no_id:
            record['id_'] = self.id_
        return record

    @property
    def info(self) -> str:
        return f"""
Name:                       {self.name}
Classes:                    {self.classes}
"""
# number of annotated ROIs:   {len(self.project.get_linked_objects('ROI', to=self))}
# number of annotations:      {len(self.project.get_linked_objects('RoiAnnotations', to=self))}
#         """
