"""
Classes defining the model and objects required to declare and manage the SAM2 prompts
"""
import sys

import numpy as np
from PySide6.QtCore import QSortFilterProxyModel, Qt, QModelIndex, QItemSelectionModel
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (QTreeView, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem, QHeaderView, QGraphicsPolygonItem,
                               QMenu)

ObjectReferenceRole = Qt.UserRole + 1


class Object:
    """
    A class defining an object that will be segmented
    """

    def __init__(self, id_: int):
        self.id_ = id_
        self.exit_frame = sys.maxsize


class BoundingBox:
    """
    A class defining a bounding box with its properties and available methods
    """

    def __init__(self, name: str = None, box: QGraphicsRectItem = None, frame: int = None, obj: Object = None):
        self.name = name
        self.graphics_item = box
        self.frame = frame
        self.object = obj

    @property
    def x(self) -> float | None:
        """
        the x coordinate of the bounding box (top-left corner)
        """
        if self.graphics_item is not None:
            return self.graphics_item.pos().x()
        return None

    @property
    def y(self) -> float | None:
        """
        the y coordinate of the bounding box (top-left corner)
        """
        if self.graphics_item is not None:
            return self.graphics_item.pos().y()
        return None

    @property
    def width(self) -> int | None:
        """
        the width of the bounding box
        """
        if self.graphics_item is not None:
            return self.graphics_item.rect().width()
        return None

    @property
    def height(self) -> int | None:
        """
        the height of the bounding box
        """
        if self.graphics_item is not None:
            return self.graphics_item.rect().height()
        return None

    @property
    def coords(self) -> list[float]:
        """
        the coordinates of the bounding box (top-left corner / bottom-right corner)
        """
        if self.x is None:
            return []
        return [self.x, self.y, self.x + self.width, self.y + self.height]

    def change_box(self, box: QGraphicsRectItem) -> None:
        """
        changes the specified bounding box
        :param box: the new graphics item for the bounding box
        """
        self.graphics_item = box
        if box is None:
            self.name = None
        else:
            self.name = box.data(0)

    def __repr__(self) -> str:
        """
        returns a representation of the bounding box (name and corresponding graphics item)
        """
        return f'{self.name=}, {self.graphics_item=}'


class Point:
    """
    A class defining a point with its properties and available methods
    """

    def __init__(self, name: str = None, point: QGraphicsEllipseItem = None, label: int = 1, frame: int = None, obj=None):
        self.name = name
        self.graphics_item = point
        self.label = label
        self.frame = frame
        self.object = obj

    @property
    def x(self) -> float | None:
        """
        the x coordinate of the point
        """
        if self.graphics_item is not None:
            return self.graphics_item.pos().x()
        return None

    @property
    def y(self) -> float | None:
        """
        the y coordinate of the point
        """
        if self.graphics_item is not None:
            return self.graphics_item.pos().y()
        return None

    @property
    def coords(self) -> list[float]:
        """
        the coordinates of the point
        """
        if self.x is None:
            return []
        return [self.x, self.y]

    # def change_point(self, point):
    #     """
    #     change the point
    #     :param point:
    #     """
    #     self.graphics_item = point
    #     if point is None:
    #         self.name = None
    #     else:
    #         self.name = point.data(0)

    def __repr__(self):
        """
        returns a representation of the point (name and corresponding graphics item)
        """
        return f'{self.name=}, {self.graphics_item=}'


class Mask:
    def __init__(self, name: str = None, mask_item: QGraphicsPolygonItem = None, frame: int = None, obj=None):
        self.name = name
        self.graphics_item = mask_item
        self.frame = frame
        self.object = obj


class ModelItem(QStandardItem):
    """
    A class defining a model item and its references to underlying object or graphics item
    """

    def __init__(self, data, obj: Object | BoundingBox | Point | Mask | None):
        super().__init__(data)
        self.setData(obj, ObjectReferenceRole)

    @property
    def object(self) -> Object | BoundingBox | Point:
        """
        returns the object corresponding to the model item
        :return:
        """
        obj = self.data(ObjectReferenceRole)
        if isinstance(obj, QGraphicsRectItem):
            obj = obj.parentItem().data(ObjectReferenceRole)
        return obj

    @property
    def graphics_item(self) -> QGraphicsItem | None:
        """
        the graphics item corresponding to the model item
        """
        obj = self.data(ObjectReferenceRole)
        if isinstance(obj, BoundingBox | Point):
            return obj.graphics_item
        return None

    @property
    def name(self) -> str:
        """
        the name of the model item
        """
        return self.data(0)

    def children(self, frame: int = None) -> list[QStandardItem]:
        """
        all the children of the model item. If frame is specified, only children corresponding to the requested frame are returned,
        otherwise, all children are.
        :param frame: the frame
        """
        if frame is None:
            return [self.child(row) for row in range(self.rowCount())]
        return [self.child(row) for row in range(self.rowCount()) if self.child(row, 1).data(0) == str(frame)]


class PromptSourceModel(QStandardItemModel):
    """
    The source model for the SAM2 prompt
    """

    def __init__(self):
        super().__init__(0, 7)
        self.setHorizontalHeaderLabels(['object', '', 'x', 'y', 'width', 'height', 'label'])
        self.root_item = self.invisibleRootItem()

    @property
    def objects(self) -> list[Object]:
        """
        all the objects to segment that were declared in the source model
        """
        return [item.object for item in self.object_items()]

    def object(self, obj_id):
        return next((obj for obj in self.objects if obj.id_ == obj_id), None)

    def object_items(self) -> list[ModelItem]:
        """
        all the model items for objects to segment that were declared in the source model
        """
        return [self.root_item.child(row) for row in range(self.root_item.rowCount())]

    def object_item(self, obj: Object) -> ModelItem:
        """
        the model item for the specified object to segment
        """
        return next((model_item for model_item in self.object_items() if model_item.object == obj), None)

    def is_present_at_frame(self, obj: Object, frame: int) -> bool:
        key_frames = self.key_frames(obj)
        if key_frames:
            return key_frames[0] <= frame < obj.exit_frame
        return True

    def get_presence_interval(self, obj: Object) -> tuple[int, int]:
        key_frames = self.key_frames(obj)
        return key_frames[0], obj.exit_frame

    def get_entry_frame(self, obj: Object) -> int:
        key_frames = self.key_frames(obj)
        return key_frames[0]

    def has_bounding_box(self, obj: Object, frame: int) -> bool:
        """
        checks that the object has a bounding box in the specified frame
        :param obj: the object
        :param frame: the frame
        """
        return any((isinstance(child.object, BoundingBox) for child in self.object_item(obj).children(frame)))

    def add_object(self, obj: Object) -> ModelItem:
        """
        Add a model item to the source model, representing an object to segment (top item below the root item)
        :param obj: the object
        :return: the model item
        """
        object_item = ModelItem(f'{obj.id_}', obj)
        self.root_item.appendRow(object_item)
        return object_item

    def remove_object(self, obj: Object) -> None:
        """
        remove an object
        :param obj: the object to remove
        """
        raise NotImplementedError

    def object_exit(self, obj: Object, frame: int):
        obj.exit_frame = frame
        print(f'Exit object {obj.id_} at frame {frame}')
        self.clean_masks(obj)
        # for mask in self.get_masks(obj):
        #     if mask.frame == frame:
        #         self.remove_mask(obj, frame)
        #     if mask.frame > frame:
        #         self.object_item(obj).removeRow(self.get_mask_row(obj, mask.frame))

    def add_bounding_box(self, obj: Object, frame: int, box: QGraphicsRectItem) -> None:
        """
        Add a bounding box defined by box: QGraphicsRectItem in the frame to the specified object
        :param obj: the object
        :param frame: the frame
        :param box: the QGraphicsRectItem
        """
        row = self.create_bounding_box_row(frame, box, obj=obj)
        self.object_item(obj).appendRow(row)

    @staticmethod
    def create_bounding_box_row(frame: int, box: QGraphicsRectItem, obj: Object = None) -> list[ModelItem | QStandardItem]:
        """
        Create a row of model items for a bounding box, to insert into the model.
        :param obj:
        :param frame: the frame
        :param box: the bounding box
        :return: the row as a list of items
        """
        bounding_box = BoundingBox(name=box.data(0), box=box, frame=frame, obj=obj)
        box_item = ModelItem(bounding_box.name, bounding_box)
        frame_item = QStandardItem(str(frame))
        x_item = QStandardItem(f'{bounding_box.x:.1f}')
        y_item = QStandardItem(f'{bounding_box.y:.1f}')
        width_item = QStandardItem(f'{int(bounding_box.width)}')
        height_item = QStandardItem(f'{int(bounding_box.height)}')
        return [box_item, frame_item, x_item, y_item, width_item, height_item]

    def change_bounding_box(self, obj: Object, frame: int, box: QGraphicsRectItem) -> None:
        """
        change the QGraphicsRectItem of the bounding box for the specified object in the specified frame
        :param obj: the object
        :param frame: the frame
        :param box: the new bounding box
        """
        bounding_box = self.get_bounding_box(obj, frame)
        if bounding_box is not None:
            bounding_box.graphics_item.scene().removeItem(bounding_box.graphics_item)
            row = self.create_bounding_box_row(frame, box, obj=obj)
            for column, item in enumerate(row):
                self.object_item(obj).setChild(self.get_bounding_box_row(obj, frame), column, item)
        else:
            self.add_bounding_box(obj, frame, box)

    def get_bounding_box(self, obj: Object, frame: int) -> BoundingBox | None:
        """
        retrieve the bounding for object in specified frame
        :param obj: the object
        :param frame: the frame
        :return: the BoundingBox object
        """
        return next((child.object for child in self.object_item(obj).children(frame) if isinstance(child.object, BoundingBox)),
                    None)

    def get_bounding_box_row(self, obj: Object, frame: int) -> int:
        """
        retrieve the row for the bounding box of object in specified frame
        :param obj: the object
        :param frame: the frame
        :return: the row number
        """
        return next((child.row() for child in self.object_item(obj).children(frame) if isinstance(child.object, BoundingBox)),
                    None)

    def get_bounding_boxes(self, obj: Object) -> list[BoundingBox]:
        """
        retrieve the bounding boxes of an object for all frame
        :param obj: the object
        :return: the list of all bounding boxes
        """
        return [child.object for child in self.object_item(obj).children() if isinstance(child.object, BoundingBox)]

    def get_bounding_box_items(self, obj: Object) -> list[ModelItem]:
        """
        retrieve the model items for bounding boxes of an object for all frame
        :param obj: the object
        :return: the list of model items
        """
        return [child for child in self.object_item(obj).children() if isinstance(child.object, BoundingBox)]

    def remove_bounding_box(self, obj: Object, frame: int) -> None:
        """
        remove a bounding box from an object in the specified frame
        :param obj: the object
        :param frame: the frame
        """
        bounding_box = self.get_bounding_box(obj, frame)
        if bounding_box is not None:
            bounding_box.graphics_item.scene().removeItem(bounding_box.graphics_item)
            self.object_item(obj).removeRow(self.get_bounding_box_row(obj, frame))
            self.clean_masks(obj)

    def remove_point(self, obj: Object, graphics_item: QGraphicsEllipseItem, frame: int = None) -> None:
        """
        remove a point from an object in the specified frame
        :param obj: the object
        :param graphics_item: the point QGraphicsEllipseItem
        :param frame: the frame
        """
        point_item = self.get_point_item_from_graphics_item(obj, graphics_item, frame)
        if point_item is not None:
            point_item.graphics_item.scene().removeItem(graphics_item)
            self.object_item(obj).removeRow(point_item.row())
            self.clean_masks(obj)

    def remove_mask(self, obj: Object, frame: int) -> None:
        """
        remove the mask from an object in the specified frame
        :param obj: the object
        :param frame: the frame
        """
        mask = self.get_mask(obj, frame)
        if mask is not None:
            if mask.graphics_item.scene() is not None:
                mask.graphics_item.scene().removeItem(mask.graphics_item)
            self.object_item(obj).removeRow(self.get_mask_row(obj, frame))

    def clean_masks(self, obj: Object):
        for mask in self.get_masks(obj):
            # if not self.is_present_at_frame(obj, mask.frame):
            if mask.frame < self.key_frames(obj)[0] or mask.frame >= obj.exit_frame:
                self.remove_mask(obj, mask.frame)

    def add_point(self, obj: Object, frame: int, point: QGraphicsEllipseItem, label: int = 1) -> None:
        """
        Add a point with a given label (1 for positive, 0 for negative) to an object in the specified frame
        :param obj: the object
        :param frame: the frame
        :param point: the point QGraphicsEllipseItem
        :param label: the label
        """
        row = self.create_point_row(frame, point, label, obj=obj)
        self.object_item(obj).appendRow(row)

    @staticmethod
    def create_point_row(frame: int, point_graphics_item: QGraphicsEllipseItem, label: int = 1, obj: Object = None) -> list[
        ModelItem | QStandardItem]:
        """
        Create a row of model items for a point, to insert into the model.
        :param obj:
        :param frame: the frame
        :param point_graphics_item: the point QGraphicsEllipseItem
        :param label: the label
        :return: the row as a list of items
        """
        point = Point(name=point_graphics_item.data(0), point=point_graphics_item, label=label, frame=frame, obj=obj)
        point_item = ModelItem(point.name, point)
        frame_item = QStandardItem(str(frame))
        x_item = QStandardItem(f'{point.x:.1f}')
        y_item = QStandardItem(f'{point.y:.1f}')
        label_item = QStandardItem(f'{label}')
        return [point_item, frame_item, x_item, y_item, QStandardItem(''), QStandardItem(''), label_item]

    def get_point_item_from_graphics_item(self, obj: Object, graphics_item: QGraphicsEllipseItem,
                                          frame: int | None = None) -> ModelItem:
        """
        retrieve the model item corresponding to a point for an object in the specified frame
        :param obj: the object
        :param graphics_item: the point QGraphicsEllipseItem
        :param frame: the frame
        :return: the model item
        """
        return next((item for item in self.object_item(obj).children(frame) if
                     item.object.graphics_item == graphics_item and not isinstance(item.object, Mask)), None)

    def get_point_from_graphics_item(self, obj: Object, graphics_item: QGraphicsEllipseItem, frame: int | None = None) -> Point:
        """
        retrieve the point for an object in the specified frame
        :param obj: the object
        :param graphics_item: the point QGraphicsEllipseItem
        :param frame: the frame
        :return: the Point object
        """
        return next((item.object for item in self.object_item(obj).children(frame) if item.object.graphics_item == graphics_item),
                    None)

    def get_points(self, obj: Object, frame: int | None = None) -> list[Point]:
        """
        retrieve the points of an object in one specific frame, or all frames (if frame is None)
        :param obj: the object
        :param frame: the frame
        :return: the list of Points
        """
        points = [child.object for child in self.object_item(obj).children(frame) if isinstance(child.object, Point)]
        if points:
            return points
        return []

    def get_point_items(self, obj: Object, frame: int | None = None) -> list[ModelItem]:
        """
        retrieve the model items for points of an object in one specific frame, or all frames (if frame is None)
        :param obj:
        :param frame:
        :return:
        """
        point_items = [child for child in self.object_item(obj).children(frame) if isinstance(child.object, Point)]
        if point_items:
            return point_items
        return []

    def set_mask(self, obj: Object, frame: int, mask: QGraphicsPolygonItem):
        current_mask = self.get_mask(obj, frame)
        if current_mask is None:
            self.add_mask(obj, frame, mask)
        else:
            current_mask.graphics_item = mask

    def add_mask(self, obj: Object, frame: int, mask: QGraphicsPolygonItem):
        row = self.create_mask_row(frame, mask, obj=obj)
        self.object_item(obj).appendRow(row)

    @staticmethod
    def create_mask_row(frame: int, mask_graphics_item: QGraphicsPolygonItem, obj: Object = None):
        mask = Mask(name=mask_graphics_item.data(0), mask_item=mask_graphics_item, frame=frame, obj=obj)
        mask_item = ModelItem(mask.name, mask)
        frame_item = QStandardItem(str(frame))
        return [mask_item, frame_item]

    def get_mask(self, obj: Object, frame: int) -> Mask | None:
        """
        retrieve the bounding for object in specified frame
        :param obj: the object
        :param frame: the frame
        :return: the BoundingBox object
        """
        return next((child.object for child in self.object_item(obj).children(frame) if isinstance(child.object, Mask)), None)

    def get_mask_row(self, obj: Object, frame: int) -> int:
        """
        retrieve the row for the mask of object in specified frame
        :param obj: the object
        :param frame: the frame
        :return: the row number
        """
        return next((child.row() for child in self.object_item(obj).children(frame) if isinstance(child.object, Mask)), None)

    def get_masks(self, obj: Object) -> list[Mask]:
        """
        retrieve the bounding boxes of an object for all frame
        :param obj: the object
        :return: the list of all bounding boxes
        """
        return [child.object for child in self.object_item(obj).children() if isinstance(child.object, Mask)]

    def get_prompt_items(self, obj: Object, frame: int | None = None) -> list[ModelItem]:
        """
        retrieve the model items for points and bounding boxes of an object in one specific frame, or all frames (if frame is None)
        :param obj: the object
        :param frame: the frame
        :return: the list of model items
        """
        prompt_items = [child for child in self.object_item(obj).children(frame) if not isinstance(child.object, Mask)]
        if prompt_items:
            return prompt_items
        return []

    def get_prompt_for_key_frame(self, obj: Object, frame: int) -> dict:
        """
        retrieve the prompt of an object for a given frame
        :param obj: the object
        :param frame: the frame
        :return: the prompt as a dictionary
        """
        box = self.get_bounding_box(obj, frame)
        points = self.get_points(obj, frame)
        if box is not None or points:
            prompt = {}
            if box is not None:
                prompt['box'] = [box.x, box.y, box.x + box.width, box.y + box.height]
            if points:
                prompt['points'] = [[point.x, point.y] for point in points]
                prompt['labels'] = [[point.label for point in points]]
        else:
            prompt = None
        return prompt

    def get_prompt_for_obj(self, obj: Object) -> dict:
        """
        retrieve the prompt for an object (all frames)
        :param obj: the object
        :return: the prompt as a dict
        """
        return {key_frame: self.get_prompt_for_key_frame(obj, key_frame) for key_frame in self.key_frames(obj)}

    def get_prompt(self) -> dict:
        """
        retrieve the prompt for all objects at all frames
        :return: the prompt as a dictionary
        """
        return {obj.id_: self.get_prompt_for_obj(obj) for obj in self.objects if self.key_frames(obj)}

    def graphics2model_item(self, graphics_item: QGraphicsRectItem | QGraphicsEllipseItem | QGraphicsPolygonItem, frame: int = None,
                            column: int = 0) -> ModelItem | QStandardItem | None:
        """
        retrieve the model item in a given column corresponding to a graphics item in the specified frame
        :param graphics_item: the graphics item
        :param frame: the frame
        :param column: the column
        :return: the model item
        """
        row_items = self.graphics2model_row(graphics_item, frame)
        if len(row_items) > column:
            return row_items[column]
        return None

    def graphics2model_row(self, graphics_item: QGraphicsRectItem | QGraphicsEllipseItem | QGraphicsPolygonItem,
                           frame: int = None) -> list[ModelItem | QStandardItem]:
        """
        retrieve the model items in the row corresponding to a graphics item in the specified frame
        :param graphics_item: the graphics item
        :param frame: the frame
        :return: the row as a list of model items
        """
        all_items = []
        for obj in self.objects:
            all_items += list(self.get_prompt_items(obj, frame))
        item = next((item for item in all_items if item.object.graphics_item == graphics_item), None)
        if item is not None:
            row = item.row()
            items_in_row = [self.itemFromIndex(self.sibling(row, col, item.index())) for col in range(self.columnCount())]
            return items_in_row
        return []

    def key_frames(self, obj: Object) -> list[int]:
        """
        retrieve the list of key frames for an object, i.e. the list of frames where there is at least one constraint (points or
        bounding box)
        :param obj: the object
        :return: the list of frames
        """
        key_frames = {self.model_item2frame(item) for item in self.get_prompt_items(obj)}
        if obj.exit_frame < sys.maxsize:
            key_frames.add(obj.exit_frame)
        return sorted(key_frames)

    @property
    def all_key_frames(self) -> list[int]:
        key_frames = set()
        for obj in self.objects:
            key_frames = key_frames.union(set(self.key_frames(obj)))
        return sorted(key_frames)

    def model_item2frame(self, item: ModelItem | QStandardItem) -> int:
        """
        retrieve the frame for a model item
        :param item: the model item
        :return: the frame number
        """
        return int(self.itemFromIndex(self.sibling(item.row(), 1, item.index())).data(0))

    def box2obj(self, box: QGraphicsRectItem) -> Object | None:
        """
        retrieve the object corresponding to the specified bounding box
        :param box: the bounding box
        :return: the object
        """
        for obj in self.objects:
            if box in [b.graphics_item for b in self.get_bounding_boxes(obj)]:
                return obj
        return None

    def point2obj(self, point: QGraphicsEllipseItem) -> Object | None:
        """
        retrieve the object corresponding to the specified point
        :param point: the point
        :return: the object
        """
        for obj in self.objects:
            if self.get_points(obj) and point in [p.graphics_item for p in self.get_points(obj)]:
                return obj
        return None

    def mask2obj(self, mask: QGraphicsPolygonItem | QGraphicsEllipseItem) -> Object | None:
        """
        retrieve the object corresponding to the specified bounding mask
        :param mask: the mask
        :return: the object
        """
        for obj in self.objects:
            if mask in [m.graphics_item for m in self.get_masks(obj)]:
                return obj
        return None

    def get_all_prompt_items(self, objects: list[Object] | None = None, frame: int | None = None) -> tuple[
        list[BoundingBox], list[Point], list[Mask]]:
        """
        retrieve all model items for bounding boxes and points in a given frame or all frames
        :param objects: the objects to retrieve prompt items for
        :param frame: the frame
        :return: the list of bounding boxes and the list of points
        """
        if objects is None:
            objects = self.objects
        boxes = [self.get_bounding_box(obj, frame) for obj in objects if self.get_bounding_box(obj, frame) is not None]
        points = []
        masks = [self.get_mask(obj, frame) for obj in objects if self.get_mask(obj, frame) is not None]
        for obj in objects:
            point_list = self.get_points(obj, frame)
            if point_list is not None:
                points += point_list
        return boxes, points, masks


class PromptProxyModel(QSortFilterProxyModel):
    """
    The proxy model to limit the views to the current frame
    """

    def __init__(self):
        super().__init__()
        self.frame = None

    def set_frame(self, frame) -> None:
        """
        Set the frame to filter model items for display in tree view
        :param frame: the frame
        """
        self.frame = frame
        self.invalidateFilter()  # Update the filter when the frame number changes

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        """
        Filter the source model to display only rows whose frame is equal to self.frame
        :param source_row: the index of the row in the source model
        :param source_parent: the index of the parent in source model
        :return:
        """
        if self.frame is None:
            return True

        model = self.sourceModel()
        index = model.index(source_row, 0, source_parent)
        item = model.itemFromIndex(index)
        if isinstance(item.object, Object):
            key_frames = model.key_frames(item.object)
            first_frame = min(key_frames) if key_frames else 0
            return first_frame <= self.frame < item.object.exit_frame
        else:
            return item.object.frame == self.frame


class ObjectsTreeView(QTreeView):
    """
    The tree view to display the objects and the prompt items
    """

    def __init__(self):
        super().__init__()
        self.source_model = None

    def contextMenuEvent(self, event):
        """
        The context menu for area manipulation

        :param event:
        """
        index = self.currentIndex()
        rect = self.visualRect(index)
        if index and rect.top() <= event.pos().y() <= rect.bottom():
            menu = QMenu()
            # view_info = menu.addAction("View info")
            # view_info.triggered.connect(lambda _: print(f'{index=}: {self.model().mapToSource(index)=}'))
            if isinstance(self.source_model.itemFromIndex(self.model().mapToSource(index)).object, Object):
                object_exit = menu.addAction("Set exit frame")
                object_exit.triggered.connect(self.object_exit)
                cancel_exit = menu.addAction("Cancel exit frame")
                cancel_exit.triggered.connect(self.cancel_exit)
            menu.exec(self.viewport().mapToGlobal(event.pos()))

    def setup(self):
        """
        Set the appearance of the tree view (hide frame column, ensure first column is wide enough to show box and point names, and
        stretch other columns to accommodate the available space.
        """
        self.setHeaderHidden(False)
        self.setColumnHidden(1, True)
        self.header().resizeSection(0, 75)
        for c in range(2, self.source_model.columnCount()):
            self.header().setSectionResizeMode(c, QHeaderView.Stretch)
        self.expandAll()

    def setSourceModel(self, model: PromptSourceModel) -> None:
        """
        Sets the source model for the view
        :param model: the source model
        """
        self.model().setSourceModel(model)
        self.source_model = model

    def object_exit(self):
        index = self.model().mapToSource(self.currentIndex())
        model_item = self.source_model.itemFromIndex(index)
        if isinstance(model_item.object, Object):
            self.source_model.object_exit(model_item.object, self.model().frame)
            self.model().invalidateFilter()

    def cancel_exit(self):
        index = self.model().mapToSource(self.currentIndex())
        model_item = self.source_model.itemFromIndex(index)
        if isinstance(model_item.object, Object):
            self.source_model.object_exit(model_item.object, sys.maxsize)
            self.model().invalidateFilter()

    def select_item(self, item) -> None:
        """
        Sets the index to the item in column 0 of the row corresponding to the given item (generally, an item from another column
        that was selected in the tree view)
        :param item: the item
        """
        self.model().invalidateFilter()
        index = self.model().mapFromSource(item.index())
        # self.selectionModel().select(index, QItemSelectionModel.ClearAndSelect)
        self.setCurrentIndex(index)

    def select_index(self, index) -> None:
        self.model().invalidateFilter()
        # self.selectionModel().select(index, QItemSelectionModel.ClearAndSelect)
        self.setCurrentIndex(index)

    def select_object_from_graphics_item(self, graphics_item: QGraphicsRectItem | QGraphicsEllipseItem | QGraphicsPolygonItem,
                                         frame: int = None) -> None:
        """
        Select the object corresponding to the selected graphics item
        :param frame: the frame
        :param graphics_item: the selected graphics item
        """
        boxes, points, masks = self.source_model.get_all_prompt_items(frame=frame)
        for prompt in boxes + points + masks:
            if graphics_item == prompt.graphics_item:
                self.select_item(self.source_model.object_item(prompt.object))
                break
        # if graphics_item in [box.graphics_item for box in boxes]:
        #     self.select_object_from_box(graphics_item)
        # elif graphics_item in [point.graphics_item for point in points]:
        #     self.select_object_from_point(graphics_item)
        # elif graphics_item in [mask.graphics_item for mask in masks]:
        #     self.select_object_from_mask(graphics_item)
        ##############################################################
        # if isinstance(graphics_item, QGraphicsRectItem):
        #     self.select_object_from_box(graphics_item)
        # elif isinstance(graphics_item, QGraphicsEllipseItem):
        #     self.select_object_from_point(graphics_item)
        # elif isinstance(graphics_item, QGraphicsPolygonItem):
        #     self.select_object_from_mask(graphics_item)

    # def select_object_from_box(self, graphics_item: QGraphicsRectItem):
    #     """
    #     Selects the object corresponding to the given bounding box
    #     :param graphics_item: the QGraphicsRectItem
    #     """
    #     self.select_item(self.source_model.object_item(self.source_model.box2obj(graphics_item)))
    #
    # def select_object_from_point(self, graphics_item: QGraphicsEllipseItem):
    #     """
    #     Selects the object corresponding to the given point
    #     :param graphics_item: the QGraphicsEllipseItem
    #     """
    #     self.select_item(self.source_model.object_item(self.source_model.point2obj(graphics_item)))
    #
    # def select_object_from_mask(self, graphics_item: QGraphicsPolygonItem | QGraphicsEllipseItem):
    #     """
    #     Selects the object corresponding to the given mask
    #     :param graphics_item: the QGraphicsPolygonItem or QGraphicsEllipseItem
    #     """
    #     self.select_item(self.source_model.object_item(self.source_model.mask2obj(graphics_item)))
