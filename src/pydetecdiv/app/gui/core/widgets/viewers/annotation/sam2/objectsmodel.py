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
from pydetecdiv.domain.ROI import ROI

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

    def __init__(self, project: Project, roi: ROI):
        super().__init__(0, 7)
        self.setHorizontalHeaderLabels(['entity', '', 'x', 'y', 'width', 'height', 'label'])
        self.root_item = self.invisibleRootItem()
        self.project = project
        self.roi = roi
        self.load_from_repository()

    def load_from_repository(self) -> None:
        """
        Load prompt model from repository database
        """
        for entity in self.roi.entities:
            self.add_entity(entity)
            self.load_bounding_boxes(entity)
            self.load_points(entity)
            self.load_masks(entity)

    def load_bounding_boxes(self, entity: Entity) -> None:
        """
        Load Entity's bounding boxes from the database repository
        :param entity: the Entity
        """
        for bounding_box in entity.bounding_boxes():
            box_item = ModelItem(bounding_box.name, bounding_box)
            frame_item = QStandardItem(str(bounding_box.frame))
            x_item = QStandardItem(f'{bounding_box.x:.1f}')
            y_item = QStandardItem(f'{bounding_box.y:.1f}')
            width_item = QStandardItem(f'{int(bounding_box.width)}')
            height_item = QStandardItem(f'{int(bounding_box.height)}')
            self.entity_item(entity).appendRow([box_item, frame_item, x_item, y_item, width_item, height_item])

    def load_points(self, entity: Entity) -> None:
        """
        Load Entity's points from the database repository
        :param entity: the Entity
        """
        for point in entity.points():
            # self.project.commit()
            point_item = ModelItem(point.name, point)
            frame_item = QStandardItem(str(point.frame))
            x_item = QStandardItem(f'{point.x:.1f}')
            y_item = QStandardItem(f'{point.y:.1f}')
            label_item = QStandardItem(f'{point.label}')
            self.entity_item(entity).appendRow(
                    [point_item, frame_item, x_item, y_item, QStandardItem(''), QStandardItem(''), label_item])

    def load_masks(self, entity: Entity) -> None:
        """
        Load Entity's Masks from the database repository
        :param entity: the Entity
        """
        for mask in entity.masks():
            mask_item = ModelItem(mask.name, mask)
            frame_item = QStandardItem(str(mask.frame))
            self.entity_item(entity).appendRow([mask_item, frame_item])

    @property
    def entities(self) -> list[Entity]:

        """
        all the objects to segment that were declared in the source model
        """
        return [item.object for item in self.entity_items()]

    def entity(self, obj_id: int) -> Entity:
        """
        return the object with the specified id
        :param obj_id: the object id
        :return: the object
        """
        return next((obj for obj in self.entities if obj.id_ == obj_id), None)

    def entity_items(self) -> list[ModelItem]:
        """
        all the model items for objects to segment that were declared in the source model
        """
        return [self.root_item.child(row) for row in range(self.root_item.rowCount())]

    def entity_item(self, entity: Entity) -> ModelItem:
        """
        the model item for the specified entity to segment
        """
        return next((model_item for model_item in self.entity_items() if model_item.object == entity), None)

    def is_present_at_frame(self, entity: Entity, frame: int) -> bool:
        """
        Returns True if the entity is present in the frame
        :param entity: the entity
        :param frame: the frame number
        :return: True if entity is in frame, False otherwise
        """
        key_frames = self.key_frames(entity)
        if key_frames:
            return key_frames[0] <= frame < entity.exit_frame
        return True

    def get_presence_interval(self, entity: Entity) -> tuple[int, int]:
        """
        Returns the time interval, in frames, that the entity is present in-frame
        :param entity: the entity
        :return: a tuple with entry frame and exit frame
        """
        key_frames = self.key_frames(entity)
        return key_frames[0], entity.exit_frame

    def get_entry_frame(self, entity: Entity) -> int:
        """
        Returns the entry frame of entity
        :param entity: the entity
        :return: the entry frame index
        """
        key_frames = self.key_frames(entity)
        return key_frames[0]

    def has_bounding_box(self, entity: Entity, frame: int) -> bool:
        """
        checks that the entity has a bounding box in the specified frame
        :param entity: the object
        :param frame: the frame
        """
        return any((isinstance(child.object, BoundingBox) for child in self.entity_item(entity).children(frame)))

    def add_entity(self, entity: Entity) -> ModelItem:
        """
        Add a model item to the source model, representing an entity to segment (top item below the root item)
        :param entity: the entity
        :return: the model item
        """
        object_item = ModelItem(f'{entity.name}', entity)
        self.root_item.appendRow(object_item)
        return object_item

    def delete_entity(self, entity: Entity) -> None:
        """
        remove an entity
        :param entity: the object to remove
        """
        bounding_boxes, points, masks = self.get_all_prompt_items([entity])
        for bounding_box in bounding_boxes:
            self.remove_bounding_box(entity, bounding_box.frame)
            del bounding_box
        for point in points:
            self.remove_point(entity, point.graphics_item)
            del point
        for mask in masks:
            self.remove_mask(entity, mask.frame)
            del mask
        self.root_item.removeRow(self.entity_item(entity).row())
        entity.delete()
        self.project.commit()
        del entity

    def entity_exit(self, entity: Entity, frame: int) -> None:
        """
        Set the status of entity at the specified frame to 'exited'
        :param entity: the entity
        :param frame: the exit frame
        """
        entity.exit_frame = frame
        print(f'Exit entity {entity.id_} at frame {frame}')
        self.clean_masks(entity)
        self.clean_prompt(entity)

    def add_bounding_box(self, entity: Entity, frame: int, box: QGraphicsRectItem) -> None:
        """
        Add a bounding box defined by box: QGraphicsRectItem in the frame to the specified entity
        :param entity: the entity
        :param frame: the frame
        :param box: the QGraphicsRectItem
        """
        row = self.create_bounding_box_row(frame, box, entity=entity)
        self.entity_item(entity).appendRow(row)

    def create_bounding_box_row(self, frame: int, box: QGraphicsRectItem, entity: Entity = None,
                                bounding_box: BoundingBox = None) -> list[ModelItem | QStandardItem]:
        """
        Create a row of model items for a bounding box, to insert into the model.
        :param entity: the entity corresponding to the bounding box
        :param frame: the frame
        :param box: the bounding box
        :return: the row as a list of items
        """
        if bounding_box is None:
            bounding_box = BoundingBox(project=self.project, name=box.data(0), box=box, frame=frame, entity=entity)
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

    def change_bounding_box(self, entity: Entity, frame: int, box: QGraphicsRectItem) -> None:
        """
        change the QGraphicsRectItem of the bounding box for the specified entity in the specified frame
        :param entity: the entity
        :param frame: the frame
        :param box: the new bounding box
        """
        # bounding_box = self.get_bounding_box(obj, frame)
        bounding_box = entity.bounding_box(frame)
        if bounding_box is not None:
            # if box != bounding_box.graphics_item:
            bounding_box.graphics_item.scene().removeItem(bounding_box.graphics_item)
            row = self.create_bounding_box_row(frame, box, entity=entity, bounding_box=bounding_box)
            for column, item in enumerate(row):
                self.entity_item(entity).setChild(self.get_bounding_box_row(entity, frame), column, item)
        else:
            self.add_bounding_box(entity, frame, box)

    def get_bounding_box_row(self, entity: Entity, frame: int) -> int:
        """
        retrieve the row for the bounding box of entity in specified frame
        :param entity: the entity
        :param frame: the frame
        :return: the row number
        """
        return next((child.row() for child in self.entity_item(entity).children(frame) if isinstance(child.object, BoundingBox)),
                    None)

    def remove_bounding_box(self, entity: Entity, frame: int) -> None:
        """
        remove a bounding box from an entity in the specified frame
        :param entity: the object
        :param frame: the frame
        """
        # bounding_box = self.get_bounding_box(obj, frame)
        bounding_box = entity.bounding_box(frame)
        if bounding_box is not None:
            if bounding_box.graphics_item.scene() is not None:
                bounding_box.graphics_item.scene().removeItem(bounding_box.graphics_item)
            self.entity_item(entity).removeRow(self.get_bounding_box_row(entity, frame))
            self.clean_masks(entity)
            self.project.delete(bounding_box)
            self.project.commit()

    def remove_point(self, entity: Entity, graphics_item: QGraphicsEllipseItem, frame: int = None) -> None:
        """
        remove a point from an entity in the specified frame
        :param entity: the entity
        :param graphics_item: the point QGraphicsEllipseItem
        :param frame: the frame
        """
        # point_item = self.get_point_item_from_graphics_item(obj, graphics_item, frame)
        point = self.get_point_from_graphics_item(entity, graphics_item, frame)
        if point is not None:
            if point.graphics_item.scene() is not None:
                point.graphics_item.scene().removeItem(graphics_item)
            self.entity_item(entity).removeRow(self.get_point_item_from_graphics_item(entity, graphics_item, frame).row())
            self.clean_masks(entity)
            self.project.delete(point)
            self.project.commit()

    def remove_mask(self, entity: Entity, frame: int) -> None:
        """
        remove the mask from an object in the specified frame
        :param obj: the object
        :param frame: the frame
        """
        # mask = self.get_mask(obj, frame)
        mask = entity.mask(frame)
        if mask is not None:
            if mask.graphics_item.scene() is not None:
                mask.graphics_item.scene().removeItem(mask.graphics_item)
            if mask.ellipse_item.scene() is not None:
                mask.ellipse_item.scene().removeItem(mask.ellipse_item)
            self.entity_item(entity).removeRow(self.get_mask_row(entity, frame))
            self.project.delete(mask)
            self.project.commit()

    def clean_masks(self, obj: Entity) -> None:
        """
        Cleans the masks for object, essentially used to remove masks that were predicted after the exit frame before it was defined
        by the user

        :param obj: the object
        """
        # for mask in self.get_masks(obj):
        for mask in obj.masks():
            # if not self.is_present_at_frame(obj, mask.frame):
            if mask.frame < self.key_frames(obj)[0] or mask.frame >= obj.exit_frame:
                self.remove_mask(obj, mask.frame)

    def clean_prompt(self, entity: Entity) -> None:
        """
        Removes the bounding boxes and points for entity that may have been defined in frames after the exit frame

        :param entity: the Entity
        """
        # for bounding_box in self.get_bounding_boxes(entity):
        for bounding_box in entity.bounding_boxes():
            if bounding_box.frame >= entity.exit_frame:
                self.remove_bounding_box(entity, bounding_box.frame)
        # for point in self.get_points(entity):
        for point in entity.points():
            if point.frame >= entity.exit_frame:
                self.remove_point(entity, point.graphics_item, point.frame)

    def add_point(self, entity: Entity, frame: int, point: QGraphicsEllipseItem, label: int = 1) -> None:
        """
        Add a point with a given label (1 for positive, 0 for negative) to an entity in the specified frame
        :param entity: the object
        :param frame: the frame
        :param point: the point QGraphicsEllipseItem
        :param label: the label
        """
        row = self.create_point_row(frame, point, label, entity=entity)
        self.entity_item(entity).appendRow(row)

    def create_point_row(self, frame: int, point_graphics_item: QGraphicsEllipseItem,
                         label: int = 1, entity: Entity = None) -> list[ModelItem | QStandardItem]:
        """
        Create a row of model items for a point, to insert into the model.
        :param entity:
        :param frame: the frame
        :param point_graphics_item: the point QGraphicsEllipseItem
        :param label: the label
        :return: the row as a list of items
        """
        point = Point(project=self.project, name=point_graphics_item.data(0), point=point_graphics_item, label=label, frame=frame,
                      entity=entity)
        self.project.commit()
        point_item = ModelItem(point.name, point)
        frame_item = QStandardItem(str(frame))
        x_item = QStandardItem(f'{point.x:.1f}')
        y_item = QStandardItem(f'{point.y:.1f}')
        label_item = QStandardItem(f'{label}')
        return [point_item, frame_item, x_item, y_item, QStandardItem(''), QStandardItem(''), label_item]

    def get_point_item_from_graphics_item(self, entity: Entity, graphics_item: QGraphicsEllipseItem,
                                          frame: int | None = None) -> ModelItem:
        """
        retrieve the model item corresponding to a point for an object in the specified frame
        :param obj: the object
        :param graphics_item: the point QGraphicsEllipseItem
        :param frame: the frame
        :return: the model item
        """
        return next((item for item in self.entity_item(entity).children(frame) if
                     item.object.graphics_item == graphics_item and not isinstance(item.object, Mask)), None)

    def get_point_from_graphics_item(self, entity: Entity, graphics_item: QGraphicsEllipseItem, frame: int | None = None) -> Point:
        """
        retrieve the point for an object in the specified frame
        :param obj: the object
        :param graphics_item: the point QGraphicsEllipseItem
        :param frame: the frame
        :return: the Point object
        """
        return next(
                (item.object for item in self.entity_item(entity).children(frame) if item.object.graphics_item == graphics_item),
                None)

    def set_mask(self, entity: Entity, frame: int, mask: QGraphicsPolygonItem) -> None:
        """
        Sets the mask for the entity in the specified frame, if the mask does not exist, it is created with add_mask method

        :param entity: the entity
        :param frame: the frame
        :param mask: the mask
        """
        current_mask = self.get_mask(entity, frame)
        if current_mask is None:
            self.add_mask(entity, frame, mask)
        else:
            current_mask.graphics_item = mask
            current_mask.ellipse_item = None

    def add_mask(self, entity: Entity, frame: int, mask: QGraphicsPolygonItem):
        """
        Adds a new mask to the entity in the specified frame

        :param entity: the entity
        :param frame: the frame
        :param mask: the mask
        """
        row = self.create_mask_row(frame, mask, entity=entity)
        self.entity_item(entity).appendRow(row)

    def create_mask_row(self, frame: int, mask_graphics_item: QGraphicsPolygonItem | QGraphicsEllipseItem,
                        entity: Entity = None) -> list[ModelItem | QStandardItem]:
        """
        Creates a row representing the mask for entity in frame
        :param frame: the frame
        :param mask_graphics_item: the mask graphics item
        :param entity: the entity
        :return: the model row for the mask
        """
        mask = Mask(project=self.project, name=mask_graphics_item.data(0), mask_item=mask_graphics_item, frame=frame, entity=entity)
        self.project.commit()
        mask_item = ModelItem(mask.name, mask)
        frame_item = QStandardItem(str(frame))
        return [mask_item, frame_item]

    def get_mask(self, entity: Entity, frame: int) -> Mask | None:
        """
        retrieve the bounding for entity in specified frame
        :param entity: the entity
        :param frame: the frame
        :return: the BoundingBox object
        """
        return next((child.object for child in self.entity_item(entity).children(frame) if isinstance(child.object, Mask)), None)

    def get_mask_row(self, entity: Entity, frame: int) -> int | None:
        """
        retrieve the row for the mask of object in specified frame
        :param obj: the object
        :param frame: the frame
        :return: the row number
        """
        return next((child.row() for child in self.entity_item(entity).children(frame) if isinstance(child.object, Mask)), None)

    def get_masks(self, entity: Entity) -> list[Mask]:
        """
        retrieve the bounding boxes of an object for all frame
        :param obj: the object
        :return: the list of all bounding boxes
        """
        return [child.object for child in self.entity_item(entity).children() if isinstance(child.object, Mask)]

    def get_all_masks(self):
        """
        Return the list of masks for all entities in every frame
        :return: the list of masks
        """
        frames = {}
        for entity in self.entities:
            # for mask in self.get_masks(entity):
            for mask in entity.masks():
                f = str(mask.frame)
                if f not in frames:
                    frames[f] = []
                frames[str(mask.frame)].append(mask)
        return frames

    def get_prompt_items(self, entity: Entity, frame: int | None = None) -> list[ModelItem]:
        """
        retrieve the model items for points and bounding boxes of an entity in one specific frame, or all frames (if frame is None)
        :param entity: the entity
        :param frame: the frame
        :return: the list of model items
        """
        prompt_items = [child for child in self.entity_item(entity).children(frame) if not isinstance(child.object, Mask)]
        if prompt_items:
            return prompt_items
        return []

    def get_prompt_for_key_frame(self, entity: Entity, frame: int) -> dict:
        """
        retrieve the prompt of an object for a given frame
        :param obj: the object
        :param frame: the frame
        :return: the prompt as a dictionary
        """
        # box = self.get_bounding_box(obj, frame)
        box = entity.bounding_box(frame)
        # points = self.get_points(obj, frame)
        points = entity.points(frame=frame)
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

    def get_prompt_for_entity(self, entity: Entity) -> dict:
        """
        retrieve the prompt for an object (all frames)
        :param obj: the object
        :return: the prompt as a dict
        """
        return {key_frame: self.get_prompt_for_key_frame(entity, key_frame) for key_frame in self.key_frames(entity)}

    def get_prompt(self) -> dict:
        """
        retrieve the prompt for all objects at all frames
        :return: the prompt as a dictionary
        """
        return {entity.id_: self.get_prompt_for_entity(entity) for entity in self.entities if self.key_frames(entity)}

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
        for obj in self.entities:
            all_items += list(self.get_prompt_items(obj, frame))
        item = next((item for item in all_items if item.object.graphics_item == graphics_item), None)
        if item is not None:
            row = item.row()
            items_in_row = [self.itemFromIndex(self.sibling(row, col, item.index())) for col in range(self.columnCount())]
            return items_in_row
        return []

    def key_frames(self, entity: Entity) -> list[int]:
        """
        retrieve the list of key frames for an entity, i.e. the list of frames where there is at least one constraint (points or
        bounding box)
        :param entity: the entity
        :return: the list of frames
        """
        key_frames = {self.model_item2frame(item) for item in self.get_prompt_items(entity)}
        if entity.exit_frame < sys.maxsize:
            key_frames.add(entity.exit_frame)
        return sorted(key_frames)

    @property
    def all_key_frames(self) -> list[int]:
        """
        All the key frames for the current video
        :return: a sorted list of key frames
        """
        key_frames = set()
        for entity in self.entities:
            key_frames = key_frames.union(set(self.key_frames(entity)))
        return sorted(key_frames)

    def model_item2frame(self, item: ModelItem | QStandardItem) -> int:
        """
        retrieve the frame for a model item
        :param item: the model item
        :return: the frame number
        """
        return int(self.itemFromIndex(self.sibling(item.row(), 1, item.index())).data(0))


    def get_all_prompt_items(self, entities: list[Entity] | None = None, frame: int | None = None) -> tuple[
        list[BoundingBox], list[Point], list[Mask]]:
        """
        retrieve all model items for bounding boxes and points in a given frame or all frames
        :param entities: the entities to retrieve prompt items for
        :param frame: the frame
        :return: the list of bounding boxes and the list of points
        """
        if entities is None:
            entities = self.entities
        # boxes = [self.get_bounding_box(obj, frame) for obj in objects if self.get_bounding_box(obj, frame) is not None]
        boxes = []
        points = []
        # masks = [self.get_mask(obj, frame) for obj in objects if self.get_mask(obj, frame) is not None]
        masks = []
        for entity in entities:
            boxes += entity.bounding_boxes(frame=frame)
            points += entity.points(frame=frame)
            masks += entity.masks(frame=frame)
        return boxes, points, masks

    def update_Item(self, graphics_item: QGraphicsRectItem | QGraphicsEllipseItem, frame: int) -> None:
        """
        Update the data for a prompt item (position and size of bounding box, position of points, ...) and save it in database
        :param graphics_item: the modified graphics item
        :param frame: the frame
        """
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
                entity_exit = menu.addAction("Set exit frame")
                entity_exit.triggered.connect(self.entity_exit)
                cancel_exit = menu.addAction("Cancel exit frame")
                cancel_exit.triggered.connect(self.cancel_exit)
                delete_entity = menu.addAction("Delete entity")
                delete_entity.triggered.connect(self.delete_entity)
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

    def delete_entity(self) -> None:
        """
        Delete the object selected in the Tree View
        """
        index = self.model().mapToSource(self.currentIndex())
        model_item = self.source_model.itemFromIndex(index)
        if isinstance(model_item.object, Entity):
            self.source_model.delete_entity(model_item.object)
            self.model().invalidateFilter()

    def entity_exit(self) -> None:
        """
        Specifies that the object selected in the tree view is not in the current frame any more
        """
        index = self.model().mapToSource(self.currentIndex())
        model_item = self.source_model.itemFromIndex(index)
        if isinstance(model_item.object, Entity):
            self.source_model.entity_exit(model_item.object, self.model().frame)
            self.model().invalidateFilter()

    def cancel_exit(self) -> None:
        """
        Cancels the definition of exit frame for the object selected in the tree view by setting the value of the exit frame to the
        maximum integer value
        """
        index = self.model().mapToSource(self.currentIndex())
        model_item = self.source_model.itemFromIndex(index)
        if isinstance(model_item.object, Entity):
            self.source_model.entity_exit(model_item.object, sys.maxsize)
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

    def select_entity_from_graphics_item(self, graphics_item: QGraphicsRectItem | QGraphicsEllipseItem | QGraphicsPolygonItem,
                                         frame: int = None) -> None:
        """
        Select the entity corresponding to the selected graphics item
        :param frame: the frame
        :param graphics_item: the selected graphics item
        """
        boxes, points, masks = self.source_model.get_all_prompt_items(frame=frame)
        for prompt in boxes + points + masks:
            if graphics_item == prompt.graphics_item:
                self.select_item(self.source_model.entity_item(prompt.object))
                break
