from PySide6.QtWidgets import QTreeView, QGraphicsRectItem

from pydetecdiv.app.models.Trees import TreeModel, TreeItem


class BoundingBox:
    def __init__(self, name: str = None, box: QGraphicsRectItem = None, item: TreeItem = None):
        self.name = name
        self.rect_item = box
        self.item = item

    def change_box(self, box):
        self.rect_item = box
        self.name = box.data(0)
        if self.item is None:
            self.item = TreeItem([self.name, self.rect_item])
        else:
            self.item.set_data(0, self.name)

    def __repr__(self):
        return f'{self.name=}, {self.rect_item=}, {self.item=}'


class SAM2prompt:
    def __init__(self, frame: int):
        self.frame = frame
        self.box = BoundingBox()
        self.points = []
        self.labels = []

    def set_bounding_box(self, box: QGraphicsRectItem) -> BoundingBox:
        if self.box.name is not None:
            box.scene().delete_item(self.box.rect_item)
        self.box.change_box(box)
        return self.box


class Object:
    def __init__(self, id_: int):
        self.id_ = id_
        self._prompt: list[SAM2prompt] = []
        self.tree_item = TreeItem([id_, self])

    def prompt(self, frame: int) -> SAM2prompt:
        return next((prompt for prompt in self._prompt if prompt.frame == frame), None)

    def set_bounding_box(self, frame: int, box: QGraphicsRectItem) -> BoundingBox:
        if self.prompt(frame) is None:
            self._prompt.append(SAM2prompt(frame))
        return self.prompt(frame).set_bounding_box(box)


class ObjectTreeModel(TreeModel):
    def __init__(self):
        super().__init__(['Objects to segment'])
        self.all_items = set()

    def add_object(self, obj: Object):
        return self.add_item(self.root_item, obj.tree_item)

    def add_item(self, parent_item: TreeItem, new_item: TreeItem):
        if new_item not in self.all_items:
            parent_item.append_child(new_item)
            self.all_items.add(new_item)
        self.layoutChanged.emit()
        return new_item

    def set_bounding_box(self, frame: int, obj: Object, box: QGraphicsRectItem):
        bounding_box = obj.set_bounding_box(frame, box)
        self.add_item(obj.tree_item, bounding_box.item)
        self.layoutChanged.emit()

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
