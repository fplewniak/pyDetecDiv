"""
Classes defining the model and objects required to declare and manage the SAM2 prompts
"""
import sys

from PySide6.QtCore import QSortFilterProxyModel, Qt, QModelIndex
from PySide6.QtGui import QStandardItemModel, QStandardItem, QContextMenuEvent
from PySide6.QtWidgets import (QTreeView, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem, QHeaderView, QGraphicsPolygonItem,
                               QMenu)

from pydetecdiv.domain.BoundingBox import BoundingBox
from pydetecdiv.domain.Entity import Entity
from pydetecdiv.domain.Mask import Mask
from pydetecdiv.domain.Point import Point
from pydetecdiv.domain.Project import Project

ObjectReferenceRole = Qt.UserRole + 1


class ModelItem(QStandardItem):
    """
    A class defining a model item and its references to underlying object or graphics item
    """

    def __init__(self, data, obj: Entity | BoundingBox | Point | Mask | None):
        super().__init__(data)
        self.setData(obj, ObjectReferenceRole)

    @property
    def object(self) -> Entity | BoundingBox | Point | Mask:
        """
        returns the object corresponding to the model item
        :return:
        """
        obj = self.data(ObjectReferenceRole)
        # if isinstance(obj, QGraphicsRectItem):
        #     obj = obj.parentItem().data(ObjectReferenceRole)
        return obj

    @property
    def graphics_item(self) -> QGraphicsItem | None:
        """
        the graphics item corresponding to the model item
        """
        obj = self.data(ObjectReferenceRole)
        if isinstance(obj, (BoundingBox, Point, Mask)):
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

    def __init__(self, project: Project):
        super().__init__(0, 7)
        self.setHorizontalHeaderLabels(['entity', '', 'x', 'y', 'width', 'height', 'label'])
        self.root_item = self.invisibleRootItem()
        self.project = project

    @property
    def objects(self) -> list[Entity]:
        """
        all the objects to segment that were declared in the source model
        """
        return [item.object for item in self.object_items()]

    def object(self, obj_id: int) -> Entity:
        """
        return the object with the specified id
        :param obj_id: the object id
        :return: the object
        """
        return next((obj for obj in self.objects if obj.id_ == obj_id), None)

    def object_items(self) -> list[ModelItem]:
        """
        all the model items for objects to segment that were declared in the source model
        """
        return [self.root_item.child(row) for row in range(self.root_item.rowCount())]

    def object_item(self, obj: Entity) -> ModelItem:
        """
        the model item for the specified object to segment
        """
        return next((model_item for model_item in self.object_items() if model_item.object == obj), None)

    def is_present_at_frame(self, obj: Entity, frame: int) -> bool:
        """
        Returns True if the object is present in the frame
        :param obj: the Object
        :param frame: the frame number
        :return: True if object is in frame, False otherwise
        """
        key_frames = self.key_frames(obj)
        if key_frames:
            return key_frames[0] <= frame < obj.exit_frame
        return True

    def get_presence_interval(self, obj: Entity) -> tuple[int, int]:
        """
        Returns the time interval, in frames, that the object is present in-frame
        :param obj: the object
        :return: a tuple with entry frame and exit frame
        """
        key_frames = self.key_frames(obj)
        return key_frames[0], obj.exit_frame

    def get_entry_frame(self, obj: Entity) -> int:
        """
        Returns the entry frame of object
        :param obj: the object
        :return: the entry frame index
        """
        key_frames = self.key_frames(obj)
        return key_frames[0]

    def has_bounding_box(self, obj: Entity, frame: int) -> bool:
        """
        checks that the object has a bounding box in the specified frame
        :param obj: the object
        :param frame: the frame
        """
        return any((isinstance(child.object, BoundingBox) for child in self.object_item(obj).children(frame)))

    def add_object(self, obj: Entity) -> ModelItem:
        """
        Add a model item to the source model, representing an object to segment (top item below the root item)
        :param obj: the object
        :return: the model item
        """
        object_item = ModelItem(f'{obj.name}', obj)
        self.root_item.appendRow(object_item)
        return object_item

    def delete_object(self, obj: Entity) -> None:
        """
        remove an object
        :param obj: the object to remove
        """
        bounding_boxes, points, masks = self.get_all_prompt_items([obj])
        # print(f'{len(bounding_boxes)}: {bounding_boxes=}')
        for mask in masks:
            self.remove_mask(obj, mask.frame)
            # del mask.out_mask
            # del mask.graphics_item
            del mask
        for bounding_box in bounding_boxes:
            self.remove_bounding_box(obj, bounding_box.frame)
            del bounding_box
        for point in points:
            self.remove_point(obj, point.graphics_item)
            del point

        for box in self.get_bounding_boxes(obj):
            if box.graphics_item.scene() is not None:
                box.graphics_item.scene().removeItem(box.graphics_item)
                del box.graphics_item
                del box

        for point in self.get_points(obj):
            if point.graphics_item.scene() is not None:
                point.graphics_item.scene().removeItem(point.graphics_item)
                del point.graphics_item
                del point
        self.root_item.removeRow(self.object_item(obj).row())
        obj.delete()
        del obj

    def object_exit(self, obj: Entity, frame: int) -> None:
        """
        Set the status of obj at the specified frame to exited
        :param obj: the object
        :param frame: the exit frame
        """
        obj.exit_frame = frame
        print(f'Exit object {obj.id_} at frame {frame}')
        self.clean_masks(obj)

    def add_bounding_box(self, obj: Entity, frame: int, box: QGraphicsRectItem) -> None:
        """
        Add a bounding box defined by box: QGraphicsRectItem in the frame to the specified object
        :param obj: the object
        :param frame: the frame
        :param box: the QGraphicsRectItem
        """
        row = self.create_bounding_box_row(frame, box, obj=obj)
        self.object_item(obj).appendRow(row)

    def create_bounding_box_row(self, frame: int, box: QGraphicsRectItem, obj: Entity = None,
                                bounding_box: BoundingBox = None) -> list[ModelItem | QStandardItem]:
        """
        Create a row of model items for a bounding box, to insert into the model.
        :param obj:
        :param frame: the frame
        :param box: the bounding box
        :return: the row as a list of items
        """
        if bounding_box is None:
            bounding_box = BoundingBox(project=self.project, name=box.data(0), box=box, frame=frame, entity=obj)
            box_item = ModelItem(bounding_box.name, bounding_box)
        else:
            bounding_box.change_box(box)
            box_item = self.graphics2model_item(box, frame, 0)
        self.project.commit()
        frame_item = QStandardItem(str(frame))
        x_item = QStandardItem(f'{bounding_box.x:.1f}')
        y_item = QStandardItem(f'{bounding_box.y:.1f}')
        width_item = QStandardItem(f'{int(bounding_box.width)}')
        height_item = QStandardItem(f'{int(bounding_box.height)}')
        return [box_item, frame_item, x_item, y_item, width_item, height_item]

    def change_bounding_box(self, obj: Entity, frame: int, box: QGraphicsRectItem) -> None:
        """
        change the QGraphicsRectItem of the bounding box for the specified object in the specified frame
        :param obj: the object
        :param frame: the frame
        :param box: the new bounding box
        """
        bounding_box = self.get_bounding_box(obj, frame)
        # print(f'{obj.name=}, {frame=}: {bounding_box=} {box=}')
        if bounding_box is not None:
            # if box != bounding_box.graphics_item:
            bounding_box.graphics_item.scene().removeItem(bounding_box.graphics_item)
            row = self.create_bounding_box_row(frame, box, obj=obj, bounding_box=bounding_box)
            for column, item in enumerate(row):
                self.object_item(obj).setChild(self.get_bounding_box_row(obj, frame), column, item)
        else:
            self.add_bounding_box(obj, frame, box)

    def get_bounding_box(self, obj: Entity, frame: int) -> BoundingBox | None:
        """
        retrieve the bounding for object in specified frame
        :param obj: the object
        :param frame: the frame
        :return: the BoundingBox object
        """
        return next((child.object for child in self.object_item(obj).children(frame) if isinstance(child.object, BoundingBox)),
                    None)

    def get_bounding_box_row(self, obj: Entity, frame: int) -> int:
        """
        retrieve the row for the bounding box of object in specified frame
        :param obj: the object
        :param frame: the frame
        :return: the row number
        """
        return next((child.row() for child in self.object_item(obj).children(frame) if isinstance(child.object, BoundingBox)),
                    None)

    def get_bounding_boxes(self, obj: Entity) -> list[BoundingBox]:
        """
        retrieve the bounding boxes of an object for all frame
        :param obj: the object
        :return: the list of all bounding boxes
        """
        return [child.object for child in self.object_item(obj).children() if isinstance(child.object, BoundingBox)]

    def get_bounding_box_items(self, obj: Entity) -> list[ModelItem]:
        """
        retrieve the model items for bounding boxes of an object for all frame
        :param obj: the object
        :return: the list of model items
        """
        return [child for child in self.object_item(obj).children() if isinstance(child.object, BoundingBox)]

    def remove_bounding_box(self, obj: Entity, frame: int) -> None:
        """
        remove a bounding box from an object in the specified frame
        :param obj: the object
        :param frame: the frame
        """
        bounding_box = self.get_bounding_box(obj, frame)
        # print(f'{obj=}, {frame=}, {bounding_box=}')
        if bounding_box is not None:
            if bounding_box.graphics_item.scene() is not None:
                bounding_box.graphics_item.scene().removeItem(bounding_box.graphics_item)
            self.object_item(obj).removeRow(self.get_bounding_box_row(obj, frame))
            self.clean_masks(obj)
            self.project.delete(bounding_box)
            self.project.commit()

    def remove_point(self, obj: Entity, graphics_item: QGraphicsEllipseItem, frame: int = None) -> None:
        """
        remove a point from an object in the specified frame
        :param obj: the object
        :param graphics_item: the point QGraphicsEllipseItem
        :param frame: the frame
        """
        # point_item = self.get_point_item_from_graphics_item(obj, graphics_item, frame)
        point = self.get_point_from_graphics_item(obj, graphics_item, frame)
        if point is not None:
            if point.graphics_item.scene() is not None:
                point.graphics_item.scene().removeItem(graphics_item)
            self.object_item(obj).removeRow(self.get_point_item_from_graphics_item(obj, graphics_item, frame).row())
            self.clean_masks(obj)
            self.project.delete(point)
            self.project.commit()

    def remove_mask(self, obj: Entity, frame: int) -> None:
        """
        remove the mask from an object in the specified frame
        :param obj: the object
        :param frame: the frame
        """
        mask = self.get_mask(obj, frame)
        if mask is not None:
            if mask.graphics_item.scene() is not None:
                mask.graphics_item.scene().removeItem(mask.graphics_item)
            if mask.ellipse_item.scene() is not None:
                mask.ellipse_item.scene().removeItem(mask.ellipse_item)
            self.object_item(obj).removeRow(self.get_mask_row(obj, frame))
            self.project.delete(mask)
            self.project.commit()

    def clean_masks(self, obj: Entity) -> None:
        """
        Cleans the masks for object, essentially used to remove masks that were predicted after the exit frame before it was defined
        by the user

        :param obj: the object
        """
        for mask in self.get_masks(obj):
            # if not self.is_present_at_frame(obj, mask.frame):
            if mask.frame < self.key_frames(obj)[0] or mask.frame >= obj.exit_frame:
                self.remove_mask(obj, mask.frame)

    def add_point(self, obj: Entity, frame: int, point: QGraphicsEllipseItem, label: int = 1) -> None:
        """
        Add a point with a given label (1 for positive, 0 for negative) to an object in the specified frame
        :param obj: the object
        :param frame: the frame
        :param point: the point QGraphicsEllipseItem
        :param label: the label
        """
        row = self.create_point_row(frame, point, label, obj=obj)
        self.object_item(obj).appendRow(row)

    def create_point_row(self, frame: int, point_graphics_item: QGraphicsEllipseItem, label: int = 1, obj: Entity = None) -> list[
        ModelItem | QStandardItem]:
        """
        Create a row of model items for a point, to insert into the model.
        :param obj:
        :param frame: the frame
        :param point_graphics_item: the point QGraphicsEllipseItem
        :param label: the label
        :return: the row as a list of items
        """
        point = Point(project=self.project, name=point_graphics_item.data(0), point=point_graphics_item, label=label, frame=frame,
                      entity=obj)
        self.project.commit()
        point_item = ModelItem(point.name, point)
        frame_item = QStandardItem(str(frame))
        x_item = QStandardItem(f'{point.x:.1f}')
        y_item = QStandardItem(f'{point.y:.1f}')
        label_item = QStandardItem(f'{label}')
        return [point_item, frame_item, x_item, y_item, QStandardItem(''), QStandardItem(''), label_item]

    def get_point_item_from_graphics_item(self, obj: Entity, graphics_item: QGraphicsEllipseItem,
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

    def get_point_from_graphics_item(self, obj: Entity, graphics_item: QGraphicsEllipseItem, frame: int | None = None) -> Point:
        """
        retrieve the point for an object in the specified frame
        :param obj: the object
        :param graphics_item: the point QGraphicsEllipseItem
        :param frame: the frame
        :return: the Point object
        """
        return next((item.object for item in self.object_item(obj).children(frame) if item.object.graphics_item == graphics_item),
                    None)

    def get_points(self, obj: Entity, frame: int | None = None) -> list[Point]:
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

    def get_point_items(self, obj: Entity, frame: int | None = None) -> list[ModelItem]:
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

    def set_mask(self, obj: Entity, frame: int, mask: QGraphicsPolygonItem) -> None:
        """
        Sets the mask for the object in the specified frame, if the mask does not exist, it is created with add_mask method

        :param obj: the object
        :param frame: the frame
        :param mask: the mask
        """
        current_mask = self.get_mask(obj, frame)
        if current_mask is None:
            self.add_mask(obj, frame, mask)
        else:
            current_mask.graphics_item = mask
            current_mask.ellipse_item = None

    def add_mask(self, obj: Entity, frame: int, mask: QGraphicsPolygonItem):
        """
        Adds a new mask to the object in the specified frame

        :param obj: the object
        :param frame: the frame
        :param mask: the mask
        """
        row = self.create_mask_row(frame, mask, obj=obj)
        self.object_item(obj).appendRow(row)

    def create_mask_row(self, frame: int, mask_graphics_item: QGraphicsPolygonItem | QGraphicsEllipseItem, obj: Entity = None) -> \
    list[
        ModelItem | QStandardItem]:
        """
        Creates a row representing the mask for object in frame
        :param frame: the frame
        :param mask_graphics_item: the mask graphics item
        :param obj: the object
        :return: the model row for the mask
        """
        mask = Mask(project=self.project, name=mask_graphics_item.data(0), mask_item=mask_graphics_item, frame=frame, entity=obj)
        self.project.commit()
        mask_item = ModelItem(mask.name, mask)
        frame_item = QStandardItem(str(frame))
        return [mask_item, frame_item]

    def get_mask(self, obj: Entity, frame: int) -> Mask | None:
        """
        retrieve the bounding for object in specified frame
        :param obj: the object
        :param frame: the frame
        :return: the BoundingBox object
        """
        return next((child.object for child in self.object_item(obj).children(frame) if isinstance(child.object, Mask)), None)

    def get_mask_row(self, obj: Entity, frame: int) -> int | None:
        """
        retrieve the row for the mask of object in specified frame
        :param obj: the object
        :param frame: the frame
        :return: the row number
        """
        return next((child.row() for child in self.object_item(obj).children(frame) if isinstance(child.object, Mask)), None)

    def get_masks(self, obj: Entity) -> list[Mask]:
        """
        retrieve the bounding boxes of an object for all frame
        :param obj: the object
        :return: the list of all bounding boxes
        """
        return [child.object for child in self.object_item(obj).children() if isinstance(child.object, Mask)]

    def get_prompt_items(self, obj: Entity, frame: int | None = None) -> list[ModelItem]:
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

    def get_prompt_for_key_frame(self, obj: Entity, frame: int) -> dict:
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

    def get_prompt_for_obj(self, obj: Entity) -> dict:
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

    def key_frames(self, obj: Entity) -> list[int]:
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
        """
        All the key frames for the current video
        :return: a sorted list of key frames
        """
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

    def box2obj(self, box: QGraphicsRectItem) -> Entity | None:
        """
        retrieve the object corresponding to the specified bounding box
        :param box: the bounding box
        :return: the object
        """
        for obj in self.objects:
            if box in [b.graphics_item for b in self.get_bounding_boxes(obj)]:
                return obj
        return None

    def point2obj(self, point: QGraphicsEllipseItem) -> Entity | None:
        """
        retrieve the object corresponding to the specified point
        :param point: the point
        :return: the object
        """
        for obj in self.objects:
            if self.get_points(obj) and point in [p.graphics_item for p in self.get_points(obj)]:
                return obj
        return None

    def mask2obj(self, mask: QGraphicsPolygonItem | QGraphicsEllipseItem) -> Entity | None:
        """
        retrieve the object corresponding to the specified bounding mask
        :param mask: the mask
        :return: the object
        """
        for obj in self.objects:
            if mask in [m.graphics_item for m in self.get_masks(obj)]:
                return obj
        return None

    def get_all_prompt_items(self, objects: list[Entity] | None = None, frame: int | None = None) -> tuple[
        list[BoundingBox], list[Point], list[Mask]]:
        """
        retrieve all model items for bounding boxes and points in a given frame or all frames
        :param objects: the objects to retrieve prompt items for
        :param frame: the frame
        :return: the list of bounding boxes and the list of points
        """
        if objects is None:
            objects = self.objects
        # boxes = [self.get_bounding_box(obj, frame) for obj in objects if self.get_bounding_box(obj, frame) is not None]
        boxes = []
        points = []
        # masks = [self.get_mask(obj, frame) for obj in objects if self.get_mask(obj, frame) is not None]
        masks = []
        for obj in objects:
            if frame is None:
                boxes_list = self.get_bounding_boxes(obj)
                masks_list = self.get_masks(obj)
            else:
                box = self.get_bounding_box(obj, frame)
                mask = self.get_mask(obj, frame)
                boxes_list = [self.get_bounding_box(obj, frame)] if box is not None else None
                masks_list = [self.get_mask(obj, frame)] if mask is not None else None
            point_list = self.get_points(obj, frame)

            if boxes_list is not None:
                boxes += boxes_list
            if point_list is not None:
                points += point_list
            if masks_list is not None:
                masks += masks_list
        return boxes, points, masks

    def update_Item(self, graphics_item: QGraphicsRectItem | QGraphicsEllipseItem, frame: int) -> None:
        model_item = self.graphics2model_item(graphics_item, frame, 0)
        if model_item is not None:
            obj = self.graphics2model_item(graphics_item, frame, 0).object
            # print(f'{obj=}')
            if isinstance(obj, BoundingBox):
                # print(f'changing box to {graphics_item} for {obj.name}')
                obj.change_box(graphics_item)
                self.project.commit()
                width_item = self.graphics2model_item(graphics_item, frame, 4)
                height_item = self.graphics2model_item(graphics_item, frame, 5)
                if width_item is not None:
                    width_item.setData(f'{int(graphics_item.rect().width())}', 0)
                if height_item is not None:
                    height_item.setData(f'{int(graphics_item.rect().height())}', 0)
            elif isinstance(obj, Point):
                obj.change_point(graphics_item)
                self.project.commit()

            x_item = self.graphics2model_item(graphics_item, frame, 2)
            y_item = self.graphics2model_item(graphics_item, frame, 3)
            if x_item is not None:
                x_item.setData(f'{graphics_item.pos().x():.1f}', 0)
            if y_item is not None:
                y_item.setData(f'{graphics_item.pos().y():.1f}', 0)


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
        if isinstance(item.object, Entity):
            key_frames = model.key_frames(item.object)
            first_frame = min(key_frames) if key_frames else 0
            return first_frame <= self.frame < item.object.exit_frame
        return item.object.frame == self.frame


class ObjectsTreeView(QTreeView):
    """
    The tree view to display the objects and the prompt items
    """

    def __init__(self):
        super().__init__()
        self.source_model = None

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """
        The context menu for area manipulation

        :param event:
        """
        index = self.currentIndex()
        rect = self.visualRect(index)
        if index and rect.top() <= event.pos().y() <= rect.bottom():
            menu = QMenu()
            if isinstance(self.source_model.itemFromIndex(self.model().mapToSource(index)).object, Entity):
                object_exit = menu.addAction("Set exit frame")
                object_exit.triggered.connect(self.object_exit)
                cancel_exit = menu.addAction("Cancel exit frame")
                cancel_exit.triggered.connect(self.cancel_exit)
                delete_object = menu.addAction("Delete object")
                delete_object.triggered.connect(self.delete_object)
            menu.exec(self.viewport().mapToGlobal(event.pos()))

    def setup(self):
        """
        Set the appearance of the tree view (hide frame column, ensure first column is wide enough to show box and point names, and
        stretch other columns to accommodate the available space.
        """
        self.setHeaderHidden(False)
        # self.setColumnHidden(1, True)
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

    def delete_object(self) -> None:
        """
        Delete the object selected in the Tree View
        """
        index = self.model().mapToSource(self.currentIndex())
        model_item = self.source_model.itemFromIndex(index)
        if isinstance(model_item.object, Entity):
            self.source_model.delete_object(model_item.object)
            self.model().invalidateFilter()

    def object_exit(self) -> None:
        """
        Specifies that the object selected in the tree view is not in the current frame any more
        """
        index = self.model().mapToSource(self.currentIndex())
        model_item = self.source_model.itemFromIndex(index)
        if isinstance(model_item.object, Entity):
            self.source_model.object_exit(model_item.object, self.model().frame)
            self.model().invalidateFilter()

    def cancel_exit(self) -> None:
        """
        Cancels the definition of exit frame for the object selected in the tree view by setting the value of the exit frame to the
        maximum integer value
        """
        index = self.model().mapToSource(self.currentIndex())
        model_item = self.source_model.itemFromIndex(index)
        if isinstance(model_item.object, Entity):
            self.source_model.object_exit(model_item.object, sys.maxsize)
            self.model().invalidateFilter()

    def select_item(self, item: ModelItem | QStandardItem) -> None:
        """
        Sets the index to the item in column 0 of the row corresponding to the given item (generally, an item from another column
        that was selected in the tree view)
        :param item: the item
        """
        self.model().invalidateFilter()
        index = self.model().mapFromSource(item.index())
        self.setCurrentIndex(index)

    def select_index(self, index: QModelIndex) -> None:
        """
        Selects the specified index
        :param index: the index in the proxy model
        """
        self.model().invalidateFilter()
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
