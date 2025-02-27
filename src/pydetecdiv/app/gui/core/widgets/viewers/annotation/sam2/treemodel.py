from PySide6.QtWidgets import QTreeView

from pydetecdiv.app.models.Trees import TreeModel


class ObjectTreeModel(TreeModel):
    def __init__(self):
        super().__init__(['Objects to segment'])

    def reset_model(self, scene):
        ...

    def update_model(self, scene):
        ...

    def delete_item(self, item):
        ...

    def delete_object(self, obj):
        ...

class ObjectsTreeView(QTreeView):
    ...
