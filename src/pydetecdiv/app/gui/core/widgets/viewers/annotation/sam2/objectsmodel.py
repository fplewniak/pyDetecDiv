from pprint import pprint

import numpy as np
from PySide6.QtCore import QSortFilterProxyModel, Qt, QModelIndex
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QTreeView, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem

ObjectReferenceRole = Qt.UserRole + 1


class BoundingBox:
    def __init__(self, name: str = None, box: QGraphicsRectItem = None):
        self.name = name
        self.graphics_item = box

    @property
    def x(self):
        if self.graphics_item is not None:
            return self.graphics_item.pos().x()
        return None

    @property
    def y(self):
        if self.graphics_item is not None:
            return self.graphics_item.pos().y()
        return None

    @property
    def width(self):
        if self.graphics_item is not None:
            return self.graphics_item.rect().width()
        return None

    @property
    def height(self):
        if self.graphics_item is not None:
            return self.graphics_item.rect().height()
        return None

    @property
    def coords(self):
        if self.x is None:
            return []
        return [self.x, self.y, self.x + self.width, self.y + self.height]

    def change_box(self, box):
        self.graphics_item = box
        if box is None:
            self.name = None
        else:
            self.name = box.data(0)

    def __repr__(self):
        return f'{self.name=}, {self.graphics_item=}'


class Point:
    def __init__(self, name: str = None, point: QGraphicsEllipseItem = None, label: int = 1):
        self.name = name
        self.graphics_item = point
        self.label = label

    @property
    def x(self):
        if self.graphics_item is not None:
            return self.graphics_item.pos().x()
        return None

    @property
    def y(self):
        if self.graphics_item is not None:
            return self.graphics_item.pos().y()
        return None

    @property
    def coords(self):
        if self.x is None:
            return []
        return [self.x, self.y]

    def change_point(self, point):
        self.graphics_item = point
        if point is None:
            self.name = None
        else:
            self.name = point.data(0)

    def __repr__(self):
        return f'{self.name=}, {self.graphics_item=}'


class Object:
    def __init__(self, id_: int):
        self.id_ = id_


class ModelItem(QStandardItem):
    def __init__(self, data, obj: Object | BoundingBox | Point | None):
        super().__init__(data)
        self.setData(obj, ObjectReferenceRole)

    @property
    def object(self):
        obj = self.data(ObjectReferenceRole)
        if isinstance(obj, QGraphicsRectItem):
            obj = obj.parentItem().data(ObjectReferenceRole)
        return obj

    @property
    def graphics_item(self):
        obj = self.data(ObjectReferenceRole)
        if isinstance(obj, QGraphicsItem):
            return obj
        return None

    @property
    def name(self):
        return self.data(0)

    def children(self, frame: int = None):
        if frame is None:
            return [self.child(row) for row in range(self.rowCount())]
        return [self.child(row) for row in range(self.rowCount()) if self.child(row, 1).data(0) == str(frame)]


class PromptSourceModel(QStandardItemModel):
    def __init__(self):
        super().__init__()
        # self.setHorizontalHeaderLabels(['object', '', 'coordinates', 'label'])
        self.setHorizontalHeaderLabels(['object', '', 'x', 'y', 'width', 'height', 'label'])
        self.root_item = self.invisibleRootItem()

    @property
    def objects(self):
        return [item.object for item in self.object_items()]

    def object_items(self):
        return [self.root_item.child(row) for row in range(self.root_item.rowCount())]

    def object_item(self, obj: Object) -> ModelItem:
        return next((model_item for model_item in self.object_items() if model_item.object == obj), None)

    def has_bounding_box(self, obj: Object, frame: int):
        return any([isinstance(child.object, BoundingBox) for child in self.object_item(obj).children(frame)])

    def add_object(self, obj: Object):
        object_item = ModelItem(f'{obj.id_}', obj)
        self.root_item.appendRow(object_item)
        return object_item

    def remove_object(self, obj: Object):
        self.objects.remove(obj)
        self.show()

    def add_bounding_box(self, obj: Object, frame: int, box: QGraphicsRectItem):
        row = self.create_bounding_box_row(frame, box)
        self.object_item(obj).appendRow(row)

    def create_bounding_box_row(self, frame: int, box: QGraphicsRectItem):
        bounding_box = BoundingBox(name=box.data(0), box=box)
        box_item = ModelItem(bounding_box.name, bounding_box)
        frame_item = QStandardItem(str(frame))
        x_item = QStandardItem(f'{bounding_box.x:.1f}')
        y_item = QStandardItem(f'{bounding_box.y:.1f}')
        width_item = QStandardItem(f'{int(bounding_box.width)}')
        height_item = QStandardItem(f'{int(bounding_box.height)}')
        return [box_item, frame_item, x_item, y_item, width_item, height_item]

    def change_bounding_box(self, obj: Object, frame: int, box: QGraphicsRectItem):
        bounding_box = self.get_bounding_box(obj, frame)
        if bounding_box is not None:
            bounding_box.graphics_item.scene().removeItem(bounding_box.graphics_item)
            row = self.create_bounding_box_row(frame, box)
            for column, item in enumerate(row):
                self.object_item(obj).setChild(self.get_bounding_box_row(obj, frame), column, row[column])
        else:
            self.add_bounding_box(obj, frame, box)

    def get_bounding_box(self, obj: Object, frame: int):
        return next((child.object for child in self.object_item(obj).children(frame) if isinstance(child.object, BoundingBox)),
                    None)

    def get_bounding_box_row(self, obj: Object, frame: int):
        return next((child.row() for child in self.object_item(obj).children(frame) if isinstance(child.object, BoundingBox)),
                    None)

    def get_bounding_boxes(self, obj: Object):
        return [child.object for child in self.object_item(obj).children() if isinstance(child.object, BoundingBox)]

    def get_bounding_box_items(self, obj: Object):
        return [child for child in self.object_item(obj).children() if isinstance(child.object, BoundingBox)]

    def remove_bounding_box(self, obj: Object, frame: int):
        bounding_box = self.get_bounding_box(obj, frame)
        if bounding_box is not None:
            bounding_box.graphics_item.scene().removeItem(bounding_box.graphics_item)
            self.object_item(obj).removeRow(self.get_bounding_box_row(obj, frame))

    def remove_point(self, obj: Object, graphics_item: QGraphicsEllipseItem, frame: int = None):
        point_item = self.get_point_from_graphics_item(obj, graphics_item, frame)
        if point_item is not None:
            point_item.object.graphics_item.scene().removeItem(graphics_item)
            self.object_item(obj).removeRow(point_item.row())

    def add_point(self, obj: Object, frame: int, point: QGraphicsEllipseItem, label: int = 1):
        row = self.create_point_row(frame, point, label)
        self.object_item(obj).appendRow(row)

    def create_point_row(self, frame: int, point_graphics_item: QGraphicsEllipseItem, label: int = 1):
        point = Point(name=point_graphics_item.data(0), point=point_graphics_item, label=label)
        point_item = ModelItem(point.name, point)
        frame_item = QStandardItem(str(frame))
        x_item = QStandardItem(f'{point.x:.1f}')
        y_item = QStandardItem(f'{point.y:.1f}')
        label_item = QStandardItem(f'{label}')
        return [point_item, frame_item, x_item, y_item, QStandardItem(''), QStandardItem(''), label_item]

    def get_point_from_graphics_item(self, obj: Object, graphics_item: QGraphicsEllipseItem, frame: int | None = None):
        return next((item for item in self.object_item(obj).children(frame) if item.object.graphics_item == graphics_item), None)

    def get_points(self, obj: Object, frame: int | None = None):
        points = [child.object for child in self.object_item(obj).children(frame) if isinstance(child.object, Point)]
        if points:
            return points
        return []

    def get_point_items(self, obj: Object, frame: int | None = None):
        point_items = [child for child in self.object_item(obj).children(frame) if isinstance(child.object, Point)]
        if point_items:
            return point_items
        return None

    def get_prompt_items(self, obj: Object, frame: int | None = None):
        prompt_items = [child for child in self.object_item(obj).children(frame)]
        if prompt_items:
            return prompt_items
        return []

    def get_prompt(self, obj: Object, frame: int):
        box = self.get_bounding_box(obj, frame)
        points = self.get_points(obj, frame)
        if box is not None or points:
            prompt = {frame: {}}
            if box is not None:
                prompt[frame]['box_coords'] = [box.x, box.y, box.x + box.width, box.y + box.height]
            if points:
                prompt[frame]['points_coords'] = [[point.x, point.y] for point in points]
                prompt[frame]['points_labels'] = [[point.label for point in points]]
        else:
            prompt = None
        return prompt

    def graphics2model_item(self, graphics_item: QGraphicsRectItem | QGraphicsEllipseItem, frame: int = None, column: int = 0):
        row_items = self.graphics2model_row(graphics_item, frame)
        if len(row_items) > column:
            return row_items[column]
        return None

    def graphics2model_row(self, graphics_item: QGraphicsRectItem | QGraphicsEllipseItem, frame: int = None):
        all_items = []
        for obj in self.objects:
            all_items += [item for item in self.get_prompt_items(obj, frame)]
        item = next((item for item in all_items if item.object.graphics_item == graphics_item), None)
        if item is not None:
            row = item.row()
            items_in_row = [self.itemFromIndex(self.sibling(row, col, item.index())) for col in range(self.columnCount())]
            return items_in_row
        return []

    def box2obj(self, box: QGraphicsRectItem):
        for obj in self.objects:
            if box in [b.graphics_item for b in self.get_bounding_boxes(obj)]:
                return obj
        return None

    def point2obj(self, point: QGraphicsEllipseItem):
        for obj in self.objects:
            if self.get_points(obj) and point in [p.graphics_item for p in self.get_points(obj)]:
                return obj
        return None

    def get_all_prompt_items(self, frame: int | None = None) -> tuple[list[BoundingBox], list[Point]]:
        boxes = [self.get_bounding_box(obj, frame) for obj in self.objects if self.get_bounding_box(obj, frame) is not None]
        points = []
        for obj in self.objects:
            point_list = self.get_points(obj, frame)
            if point_list is not None:
                points += point_list
        return boxes, points

    def show(self):
        for obj in [(self.root_item.child(r).name, self.root_item.child(r).object,) for r in range(self.root_item.rowCount())]:
            print(f'{obj=}')


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
        # Get the frame number
        frame = self.sourceModel().data(index, Qt.DisplayRole)

        # Check if the item belongs to the desired set
        return (frame is None) or (frame == self.frame)


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

    def select_object_from_graphics_item(self, graphics_item: QGraphicsRectItem | QGraphicsEllipseItem):
        if isinstance(graphics_item, QGraphicsRectItem):
            self.select_object_from_box(graphics_item)
        elif isinstance(graphics_item, QGraphicsEllipseItem):
            self.select_object_from_point(graphics_item)

    def select_object_from_box(self, graphics_item: QGraphicsRectItem):
        self.select_item(self.source_model.object_item(self.source_model.box2obj(graphics_item)))

    def select_object_from_point(self, graphics_item: QGraphicsEllipseItem):
        self.select_item(self.source_model.object_item(self.source_model.point2obj(graphics_item)))
