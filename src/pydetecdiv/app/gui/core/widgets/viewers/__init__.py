"""
Module defining classes for generic viewer API
"""
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, QRectF, Signal, QPointF
from PySide6.QtGui import QKeySequence, QTransform, QPen, QKeyEvent, QMouseEvent, QWheelEvent, QPainter
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem,
                               QAbstractGraphicsShapeItem, QWidget, QGraphicsSceneWheelEvent, QGraphicsSceneMouseEvent,
                               QStyleOptionGraphicsItem, QGraphicsEllipseItem)

from pydetecdiv.app import PyDetecDiv, DrawingTools
from pydetecdiv.utils import round_to_even


class GraphicsView(QGraphicsView):
    """
     A generic widget for graphics visualization (image, drawings, etc.) using layers
    """
    zoom_value_changed = Signal(int)

    def __init__(self, parent: QWidget = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.layers = []
        self.background = None
        self.scale_value = 100
        # sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        # self.setSizePolicy(sizePolicy)

    def setup(self, scene: QGraphicsScene = None) -> None:
        """
        Sets up the GraphicsView, adding a scene and a background layer

        :param scene: the scene to add to the GraphicsView. If None, then a generic Scene will be added
        """
        if scene is None:
            self.setScene(Scene())
        else:
            self.setScene(scene)
        self.background = self.addLayer(background=True)

    def zoom_set_value(self, value: float) -> None:
        """
        Sets the zoom to the desired value

        :param value: the value (integer, 100 representing 1:1)
        """
        s = value / self.scale_value
        scale = QTransform(s, 0, 0, 0, s, 0, 0, 0, 1)
        self.setTransform(scale, combine=True)
        self.scale_value = value
        self.zoom_value_changed.emit(value)

    def zoom_fit(self) -> None:
        """
        Set the zoom value to fit the image in the viewer
        """
        # self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.scale_value = int(100 * np.around(self.transform().m11(), 2))

    def _create_layer(self, background: bool = False) -> 'Layer':
        """
        Creates a layer object to add to the scene

        :param background: if True, the created layer is a background layer
        :return: the created layer
        """
        if background:
            return BackgroundLayer(self)
        return Layer(self)

    def addLayer(self, name: str = None, background: bool = False) -> 'Layer':
        """
        Adds a layer item to the Scene

        :param background: if True, the layer is a background layer
        :return: the added layer
        """
        layer = self._create_layer(background)
        if name is not None:
            layer.setData(0, name)
        self.scene().addItem(layer)
        layer.setZValue(len(self.layers))
        self.layers.append(layer)
        PyDetecDiv.app.scene_modified.emit(self.scene())
        return layer

    def move_layer(self, origin: int, destination: int) -> None:
        """
        Move layer from its current z position to another one

        :param origin: the origin z position
        :param destination: the destination z position
        """
        layer = self.layers.pop(origin)
        self.layers.insert(min(len(self.layers), max(1, destination)), layer)
        for i, l in enumerate(self.layers):
            l.zIndex = i


class Scene(QGraphicsScene):
    """
    A generic scene attached to a GraphicsView, containing graphics items, and reacting to user input
    (key press, mouse,...)
    """

    def __init__(self, parent: QWidget = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.pen = QPen(Qt.GlobalColor.cyan, 2)

    @property
    def viewer(self) -> GraphicsView:
        """
        A convenience property returning the scene's viewer

        :return: the scene's viewer (GraphicsViewer)
        """
        return self.view()

    def view(self, index: int = 0) -> GraphicsView:
        """
        A convenience method to return any view associated with the scene

        :param index: the index of the view
        :return: the requested view
        """
        return GraphicsView(self.views()[index])

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Detect when a key is pressed and perform the corresponding action:
        * QKeySequence.Delete: delete the selected item

        :param event: the key pressed event
        """
        if event.matches(QKeySequence.StandardKey.Delete):
            for r in self.selectedItems():
                self.delete_item(r)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        Detect when the left mouse button is pressed and perform the action corresponding to the currently checked
        drawing tool

        :param event: the mouse press event
        """
        if event.button() == Qt.MouseButton.LeftButton:
            match PyDetecDiv.current_drawing_tool, event.modifiers():
                case DrawingTools.Cursor, Qt.KeyboardModifier.NoModifier:
                    self.select_Item(event)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.NoModifier:
                    self.unselect_items(event)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.ControlModifier:
                    self.select_Item(event)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.ShiftModifier:
                    self.select_Item(event)
                case DrawingTools.DuplicateItem, Qt.KeyboardModifier.NoModifier:
                    self.duplicate_selected_Item(event)
                    self.select_Item(event)
                    PyDetecDiv.app.scene_modified.emit(self)
                case DrawingTools.DrawPoint, Qt.KeyboardModifier.NoModifier:
                    self.add_point(event)
                    PyDetecDiv.app.scene_modified.emit(self)
                case DrawingTools.DrawPoint, Qt.KeyboardModifier.ControlModifier:
                    self.add_point(event)
                    PyDetecDiv.app.scene_modified.emit(self)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        Detect mouse movement and apply the appropriate method according to the currently checked drawing tool and key
        modifier

        :param event: the mouse move event
        """
        if event.buttons() == Qt.MouseButton.LeftButton:
            match PyDetecDiv.current_drawing_tool, event.modifiers():
                case DrawingTools.Cursor, Qt.KeyboardModifier.NoModifier:
                    self.move_Item(event)
                case DrawingTools.Cursor, Qt.KeyboardModifier.ControlModifier:
                    self.draw_Item(event)
                    # PyDetecDiv.app.scene_modified.emit(self)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.NoModifier:
                    self.draw_Item(event)
                    PyDetecDiv.app.scene_modified.emit(self)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.ControlModifier:
                    self.move_Item(event)
                case DrawingTools.DrawRect, Qt.KeyboardModifier.ShiftModifier:
                    self.draw_Item(event)
                case DrawingTools.DuplicateItem, Qt.KeyboardModifier.NoModifier:
                    self.move_Item(event)

    def wheelEvent(self, event: QGraphicsSceneWheelEvent) -> None:
        """
        DOES NOT WORK
        Action triggered by using the mouse wheel
        """
        match event.modifiers():
            case Qt.KeyboardModifier.ControlModifier:
                self.viewer.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                if event.delta() > 0:
                    self.viewer.zoom_set_value(self.viewer.scale_value * 1.2)
                elif event.delta() < 0:
                    self.viewer.zoom_set_value(self.viewer.scale_value / 1.2)
                event.accept()
                self.viewer.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)

    def delete_item(self, graphics_item: QGraphicsItem) -> None:
        """
        remove a graphics item from the scene. The QGraphicsItem object is not deleted though and can be added back to the scene

        :param graphics_item: the graphics item to remove
        """
        PyDetecDiv.app.graphic_item_deleted.emit(graphics_item.data(0))
        self.removeItem(graphics_item)

    def unselect_items(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        Unselect all selected items

        :param event: the mouse press event
        """
        _ = [r.setSelected(False) for r in self.selectedItems()]

    def select_Item(self, event: QGraphicsSceneMouseEvent) -> None:
        """
        Select the current area/Item

        :param event: the mouse press event
        """
        _ = [r.setSelected(False) for r in self.selectedItems()]
        r = self.itemAt(event.scenePos(), self.viewer.transform())
        if r is not None:
            r.setSelected(True)
            if isinstance(r, QGraphicsRectItem):
                self.display_Item_size(r)
        return r

    def get_selected_Item(self) -> QAbstractGraphicsShapeItem | None:
        """
        Return the selected Item

        :return: the selected Item
        """
        for selection in self.selectedItems():
            if isinstance(selection, (QGraphicsRectItem, QGraphicsEllipseItem)):
                return selection
        return None

    def duplicate_selected_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem | None:
        """
        Duplicate the currently selected Item at the current mouse position

        :param event: the mouse press event
        """
        pos = event.scenePos()
        item = self.get_selected_Item()
        if item and isinstance(item, QGraphicsRectItem):
            # item = self.addRect(item.rect())
            item = item.__class__(item.rect())
            self.addItem(item)
            item.setPen(self.pen)
            w, h = item.rect().size().toTuple()
            item.setPos(QPoint(pos.x() - np.around(w / 2.0), pos.y() - np.around(h / 2.0)))
            item.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            item.setData(0, f'Region_{item.x():.1f}_{item.y():.1f}')
            item.setZValue(10)
            # self.select_Item(event)
            return item
        return None

    def get_colliding_ShapeItems(self, item: QAbstractGraphicsShapeItem) -> list[QAbstractGraphicsShapeItem]:
        """
        Retrieve all ShapeItems colliding with the item in this scene

        :param item: the item to check
        :return: a list of colliding items
        """
        return [r for r in item.collidingItems(Qt.ItemSelectionMode.IntersectsItemBoundingRect) if
                isinstance(r, QAbstractGraphicsShapeItem)]

    def move_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem | QGraphicsEllipseItem:
        """
        Move the currently selected Item if it is movable

        :param event: the mouse move event
        """
        item = self.get_selected_Item()
        if item and (item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable):
            if not (isinstance(item, QGraphicsEllipseItem) and PyDetecDiv.current_drawing_tool == DrawingTools.DuplicateItem):
                pos = event.scenePos()
                item.moveBy(pos.x() - event.lastScenePos().x(), pos.y() - event.lastScenePos().y())
        return item

    def draw_Item(self, event: QGraphicsSceneMouseEvent) -> QGraphicsRectItem:
        """
        Draw or redraw the currently selected Item if it is movable

        :param event: the mouse press event
        """
        item = self.get_selected_Item()
        pos = event.scenePos()
        if item and (item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable):
            item_pos = item.scenePos()
            w, h = np.max([round_to_even(pos.x() - item_pos.x()), 5]), np.max(
                    [round_to_even(pos.y() - item_pos.y()), 5])
            rect = QRect(0, 0, w, h)
            item.setRect(rect)
        elif PyDetecDiv.current_drawing_tool == DrawingTools.DrawRect:
            item = self.addRect(QRect(0, 0, 5, 5))
            item.setPen(self.pen)
            item.setPos(QPointF(pos.x(), pos.y()))
            item.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            item.setData(0, f'Region_{pos.x():.1f}_{pos.y():.1f}')
            item.setZValue(10)
            item.setSelected(True)
        if item:
            self.display_Item_size(item)
        return item

    def regions(self) -> list[QGraphicsRectItem]:
        """
        the list of regions in the scene
        """
        return [p for p in self.items() if isinstance(p, QGraphicsRectItem)]

    def set_Item_width(self, width: int):
        """
        Sets the width of the selected item

        :param width: the width value
        """
        item = self.get_selected_Item()
        if item and (item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable):
            rect = QRectF(0, 0, width, item.rect().height())
            item.setRect(rect)

    def set_Item_height(self, height: int):
        """
        Sets the height of the selected item

        :param height: the height value
        """
        item = self.get_selected_Item()
        if item and (item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable):
            rect = QRectF(0, 0, item.rect().width(), height)
            item.setRect(rect)

    def display_Item_size(self, item: QGraphicsRectItem) -> None:
        """
        Displays the item size in the Drawing tools palette

        :param item: the item
        """
        PyDetecDiv.main_window.drawing_tools.roi_width.setValue(item.rect().width())
        PyDetecDiv.main_window.drawing_tools.roi_height.setValue(item.rect().height())

    def add_point(self, event: QGraphicsSceneMouseEvent) -> QGraphicsEllipseItem:
        """
        Draw a point
        :param event: the mouse press event
        """
        pos = event.scenePos()
        item = QGraphicsEllipseItem(0, 0, 2, 2)
        self.addItem(item)
        item.setPen(self.pen)
        item.setPos(QPointF(pos.x(), pos.y()))
        item.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        item.setData(0, f'point_{pos.x():.1f}_{pos.y():.1f}')
        item.setZValue(10)
        item.setSelected(False)
        return item

    def points(self) -> list[QGraphicsEllipseItem]:
        """
        the list of points in the scene
        """
        return [p for p in self.items() if isinstance(p, QGraphicsEllipseItem)]

    def layers(self) -> list:
        """
        the list of layers in the scene
        """
        return [l for l in self.items() if isinstance(l, Layer)]

    def _add_item_to_dict(self, item, item_dict) -> None:
        """
        Adds an item to the item dictionary
        :param item: the item to add
        :param item_dict: the item dictionary
        """
        item_name = item.data(0)
        if item_name is not None:
            item_dict[item_name] = ''
        # item_name = item.data(0)
        # if item_name is None:
        #     item_name = 'xxx'
        #
        # # Create the entry for the current item
        # item_entry = {'item': item, 'children': {}}
        #
        # # Add the item to the dictionary
        # item_dict[item_name] = item_entry
        #
        # # Recursively add child items
        # for child_item in item.childItems():
        #     self._add_item_to_dict(child_item, item_entry['children'])

    def item_dict(self) -> dict:
        """
        Returns a dictionary with all items in a scene
        """
        item_dict = {'layers': {}, 'regions': {}, 'points': {}}
        for item in self.layers():
            self._add_item_to_dict(item, item_dict['layers'])
        for item in sorted(self.regions(), key=lambda x: x.data(0)):
            self._add_item_to_dict(item, item_dict['regions'])
        for item in sorted(self.points(), key=lambda x: x.data(0)):
            self._add_item_to_dict(item, item_dict['points'])
        return item_dict


class Layer(QGraphicsItem):
    """
    A graphics item containing sub-items in order to define layer behaviour
    """

    def __init__(self, viewer: GraphicsView, **kwargs):
        super().__init__(**kwargs)
        self.viewer = viewer

    @property
    def zIndex(self) -> int:
        """
        A convenience method returning the Z index of the Layer

        :return: the z index
        """
        return int(self.zValue())

    @zIndex.setter
    def zIndex(self, zIndex: int) -> None:
        self.setZValue(min(len(self.viewer.layers), max(1, zIndex)))

    def move_up(self) -> None:
        """
        Moves the layer up one level
        """
        self.viewer.move_layer(self.zIndex, self.zIndex + 1)

    def move_down(self) -> None:
        """
        Moves the layer down one level
        """
        self.viewer.move_layer(self.zIndex, self.zIndex - 1)

    def toggleVisibility(self) -> None:
        """
        Toggles visibility of the layer
        """
        self.setVisible(not self.isVisible())

    def addItem(self, item: QGraphicsItem) -> QGraphicsItem:
        """
        Adds an item to the layer

        :param item: the added item
        :return: the added item
        """
        item.setParentItem(self)
        return item

    def boundingRect(self) -> QRectF:
        """
        Returns the bounding rect of the layer, which contains all children

        :return: the bounding rect of the layer
        """
        if self.childItems():
            return self.childrenBoundingRect()
        return QRectF(0.0, 0.0, 0.0, 0.0)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = ...):
        """
        Paint method added to comply with the implementation of abstract class
        """
        pass


class BackgroundLayer(Layer):
    """
    A particular layer that is always at the bottom of a scene
    """

    def __init__(self, viewer: GraphicsView, **kwargs):
        super().__init__(viewer, **kwargs)
        self.setData(0, 'background')

    @property
    def zIndex(self) -> int:
        """
        A convenience method returning the Z index of the Layer. As this is a background layer, the Z index
        should always be 0

        :return: the z index
        """
        return int(self.zValue())

    @zIndex.setter
    def zIndex(self, zIndex: int) -> None:
        zIndex = 0
        self.setZValue(zIndex)
