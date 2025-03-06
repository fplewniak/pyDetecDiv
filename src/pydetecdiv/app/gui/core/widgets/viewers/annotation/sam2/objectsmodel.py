from pprint import pprint

from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QTreeView, QGraphicsRectItem, QGraphicsEllipseItem

ObjectReferenceRole = Qt.UserRole + 1


class BoundingBox:
    def __init__(self, name: str = None, box: QGraphicsRectItem = None):
        self.name = name
        self.rect_item = box

    @property
    def x(self):
        if self.rect_item is not None:
            return self.rect_item.pos().x()
        return None

    @property
    def y(self):
        if self.rect_item is not None:
            return self.rect_item.pos().y()
        return None

    @property
    def width(self):
        if self.rect_item is not None:
            return self.rect_item.rect().width()
        return None

    @property
    def height(self):
        if self.rect_item is not None:
            return self.rect_item.rect().height()
        return None

    @property
    def coords(self):
        if self.x is None:
            return []
        return [self.x, self.y, self.x + self.width, self.y + self.height]

    def change_box(self, box):
        self.rect_item = box
        if box is None:
            self.name = None
        else:
            self.name = box.data(0)

    def __repr__(self):
        return f'{self.name=}, {self.rect_item=}'


class Point:
    def __init__(self, name: str = None, point: QGraphicsEllipseItem = None, label: int = 1):
        self.name = name
        self.point_item = point
        self.label = label

    @property
    def x(self):
        if self.point_item is not None:
            return self.point_item.pos().x()
        return None

    @property
    def y(self):
        if self.point_item is not None:
            return self.point_item.pos().y()
        return None

    @property
    def coords(self):
        if self.x is None:
            return []
        return [self.x, self.y]

    def change_point(self, point):
        self.point_item = point
        if point is None:
            self.name = None
        else:
            self.name = point.data(0)

    def __repr__(self):
        return f'{self.name=}, {self.point_item=}'


class Object:
    def __init__(self, id_: int):
        self.id_ = id_
        self.prompt = None


class PromptSourceModel(QStandardItemModel):
    def __init__(self):
        super().__init__()
        self.setHorizontalHeaderLabels(['object', '', 'coordinates', 'label'])
        self.root_item = self.invisibleRootItem()
        self.objects = []

    def add_object(self, obj: Object):
        object_item = QStandardItem(f'{obj.id_}')
        object_item.setData(obj, ObjectReferenceRole)
        self.root_item.appendRow(object_item)
        self.show()
        return object_item

    def remove_object(self, obj: Object):
        self.objects.remove(obj)
        self.show()

    def add_bounding_box(self, obj: Object, frame: int, box: QGraphicsRectItem):
        # obj.set_bounding_box(frame, box)
        self.show()

    def remove_bounding_box(self, obj: Object, frame: int):
        # obj.set_bounding_box(frame, None)
        self.show()

    def box2obj(self, box: QGraphicsRectItem):
        for obj in self.objects:
            if box in [p.box.rect_item for p in obj._prompt]:
                return obj
        return None

    def show(self):
        for obj in [(self.root_item.child(r).model().index(r, 0, self.root_item.index()).data(),
                     self.root_item.child(r).model().index(r, 1, self.root_item.index()).data(),
                     self.root_item.child(r).model().index(r, 2, self.root_item.index()).data(),
                     self.root_item.child(r).model().index(r, 3, self.root_item.index()).data(),
                     ) for r in
                    range(self.root_item.rowCount())]:
            pprint(f'{obj=}')


class PromptProxyModel(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.frame = None

    def set_frame(self, frame):
        self.frame = str(frame)  # Ensure set_number is a string
        self.invalidateFilter()  # Update the filter when the set number changes

    def filterAcceptsRow(self, source_row, source_parent):
        if self.frame is None:
            return True

        # Get the index of the current row in the set number column
        index = self.sourceModel().index(source_row, 1, source_parent)
        # Get the set number
        frame = self.sourceModel().data(index, Qt.DisplayRole)

        # Check if the item belongs to the desired set
        return frame == self.frame


class ObjectsTreeView(QTreeView):
    def __init__(self):
        super().__init__()
        self.source_model = None
        # self.setColumnHidden(1, True)

        # self.source_model = PromptSourceModel()
        #
        # # Add root item to the model
        # root = self.source_model.invisibleRootItem()
        #
        # # Create a CustomProxyModel to filter grandchildren
        # self.proxy_model = PromptProxyModel()
        # self.setModel(self.proxy_model)
        #
        # self.proxy_model.setRecursiveFilteringEnabled(True)
        #
        # # Hide the set number column
        # self.setColumnHidden(1, True)
        # self.setHeaderHidden(False)
        # self.expandAll()

    # self.clicked.connect(lambda x: self.select_item(x.internalPointer()))

    def setSourceModel(self, model: QStandardItemModel):
        self.model().setSourceModel(model)
        self.source_model = model

    def select_item(self, item):
        self.setCurrentIndex(self.model().index(item.row(), 0))
