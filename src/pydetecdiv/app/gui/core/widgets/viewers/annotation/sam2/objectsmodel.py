from PySide6.QtWidgets import QTreeView, QGraphicsRectItem

from pydetecdiv.app.models.Trees import TreeModel, TreeItem


class SAM2prompt:
    def __init__(self, frame: int):
        self.frame = frame
        self.box = None
        self.points = []
        self.labels = []

    def set_bounding_box(self, box: QGraphicsRectItem):
        if self.box is not None:
            box.scene().delete_item(self.box)
        self.box = box


class Object:
    def __init__(self, id_: int):
        self.id_ = id_
        self._prompt: list[SAM2prompt] = []

    def prompt(self, frame: int) -> SAM2prompt:
        return next((prompt for prompt in self._prompt if prompt.frame == frame), None)

    def set_bounding_box(self, frame: int, box: QGraphicsRectItem) -> None:
        if self.prompt(frame) is None:
            self._prompt.append(SAM2prompt(frame))
        self.prompt(frame).set_bounding_box(box)


class ObjectTreeModel(TreeModel):
    def __init__(self):
        super().__init__(['Objects to segment'])
        self.all_items = set()

    def add_object(self, obj: Object):
        return self.add_item(self.root_item, TreeItem([obj.id_, obj]))

    def add_item(self, parent_item: TreeItem, new_item: TreeItem):
        parent_item.append_child(new_item)
        self.all_items.add(new_item)
        self.layoutChanged.emit()
        return new_item

    def reset_model(self, scene):
        ...

    def update_model(self, scene):
        ...

    def delete_item(self, item):
        ...

    def delete_object(self, obj):
        ...


class ObjectsTreeView(QTreeView):
    def select_item(self, item):
        self.setCurrentIndex(self.model().index(item.row(), 0))
