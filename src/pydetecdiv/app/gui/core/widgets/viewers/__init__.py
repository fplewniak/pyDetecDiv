"""
Module defining classes for generic viewer API
"""
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, QRectF, Signal
from PySide6.QtGui import QKeySequence, QTransform, QPen
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem, \
    QAbstractGraphicsShapeItem

from pydetecdiv.app import PyDetecDiv, DrawingTools
from pydetecdiv.utils import round_to_even


class GraphicsView(QGraphicsView):
    """
     A generic widget for graphics visualization (image, drawings, etc.) using layers
    """
    zoom_value_changed = Signal(int)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        # if scene_class is None:
        #     self.setScene(Scene())
        # else:
        #     self.setScene(scene_class())
        self.layers = []
        self.background = None
        self.scale_value = 100
        # self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        # self.setSizePolicy(sizePolicy)

    def setup(self, scene=None):
        """
        Sets up the GraphicsView, adding a scene and a background layer

        :param scene: the scene to add to the GraphicsView. If None, then a generic Scene will be added
        """
        if scene is None:
            self.setScene(Scene())
        else:
            self.setScene(scene)
        self.background = self.addLayer(background=True)

    def zoom_set_value(self, value):
        """
        Sets the zoom to the desired value

        :param value: the value (integer, 100 representing 1:1)
        """
        self.scale(value / self.scale_value, value / self.scale_value)
        self.scale_value = value
        self.zoom_value_changed.emit(value)

    def zoom_fit(self):
        """
        Set the zoom value to fit the image in the viewer
        """
        # self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        self.scale_value = int(100 * np.around(self.transform().m11(), 2))

    def _create_layer(self, background=False):
        """
        Creates a layer object to add to the scene

        :param background: if True, the created layer is a background layer
        :return: the created layer
        """
        if background:
            return BackgroundLayer(self)
        return Layer(self)

    def addLayer(self, background=False):
        """
        Adds a layer item to the Scene

        :param background: if True, the layer is a background layer
        :return: the added layer
        """
        layer = self._create_layer(background)
        self.scene().addItem(layer)
        layer.setZValue(len(self.layers))
        self.layers.append(layer)
        return layer

    def move_layer(self, origin, destination):
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

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.pen = QPen(Qt.GlobalColor.cyan, 2)

    @property
    def viewer(self):
        """
        A convenience property returning the scene's viewer

        :return: the scene's viewer (GraphicsViewer)
        """
        return self.view()

    def view(self, index=0):
        """
        A convenience method to return any view associated with the scene

        :param index: the index of the view
        :return: the requested view
        """
        return self.views()[index]

    def keyPressEvent(self, event):
        """
        Detect when a key is pressed and perform the corresponding action:
        * QKeySequence.Delete: delete the selected item

        :param event: the key pressed event
        :type event: QKeyEvent
        """
        if event.matches(QKeySequence.Delete):
            for r in self.selectedItems():
                self.removeItem(r)

    def mousePressEvent(self, event):
        """
        Detect when the left mouse button is pressed and perform the action corresponding to the currently checked
        drawing tool

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        if event.button() == Qt.LeftButton:
            match PyDetecDiv.current_drawing_tool:
                case DrawingTools.Cursor:
                    self.select_Item(event)
                case DrawingTools.DrawRect:
                    self.select_Item(event)
                case DrawingTools.DuplicateItem:
                    self.duplicate_selected_Item(event)

    def mouseMoveEvent(self, event):
        """
        Detect mouse movement and apply the appropriate method according to the currently checked drawing tool and key
        modifier

        :param event: the mouse move event
        :type event: QGraphicsSceneMouseEvent
        """
        if event.button() == Qt.NoButton:
            match PyDetecDiv.current_drawing_tool, event.modifiers():
                case DrawingTools.Cursor, Qt.NoModifier:
                    self.move_Item(event)
                case DrawingTools.Cursor, Qt.ControlModifier:
                    self.draw_Item(event)
                case DrawingTools.DrawRect, Qt.NoModifier:
                    self.draw_Item(event)
                case DrawingTools.DrawRect, Qt.ControlModifier:
                    self.move_Item(event)
                case DrawingTools.DuplicateItem, Qt.NoModifier:
                    self.move_Item(event)

    def wheelEvent(self, event):
        match event.modifiers():
            case Qt.ControlModifier:
                if event.delta() > 0:
                    self.viewer.zoom_set_value(self.viewer.scale_value * 1.2)
                elif event.delta() < 0:
                    self.viewer.zoom_set_value(self.viewer.scale_value / 1.2)

    def select_Item(self, event):
        """
        Select the current area/Item

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        # _ = [r.setSelected(False) for r in self.items()]
        _ = [r.setSelected(False) for r in self.selectedItems()]
        r = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
        if r is not None:
            r.setSelected(True)
            if hasattr(r, 'rect'):
                self.display_Item_size(r)

    def get_selected_Item(self):
        """
        Return the selected Item

        :return: the selected Item
        :rtype: QGraphicsRectItem
        """
        for selection in self.selectedItems():
            if isinstance(selection, QGraphicsItem):
                return selection
        return None

    def duplicate_selected_Item(self, event):
        """
        Duplicate the currently selected Item at the current mouse position

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        pos = event.scenePos()
        item = self.get_selected_Item()
        if item:
            # item = self.addRect(item.rect())
            item = item.__class__(item.rect())
            self.addItem(item)
            item.setPen(self.pen)
            w, h = item.rect().size().toTuple()
            item.setPos(QPoint(pos.x() - np.around(w / 2.0), pos.y() - np.around(h / 2.0)))
            item.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
            item.setData(0, f'Region{len(self.items())}')
            item.setZValue(10)
            self.select_Item(event)
            return item
        return None

    def get_colliding_ShapeItems(self, item):
        """
        Retrieve all ShapeItems colliding with the item in this scene

        :param item: the item to check
        :return: a list of colliding items
        """
        return [r for r in item.collidingItems(Qt.IntersectsItemBoundingRect) if
                isinstance(r, QAbstractGraphicsShapeItem)]

    def move_Item(self, event):
        """
        Move the currently selected Item if it is movable

        :param event: the mouse move event
        :type event: QGraphicsSceneMouseEvent
        """
        item = self.get_selected_Item()
        if item and (item.flags() & QGraphicsItem.ItemIsMovable):
            pos = event.scenePos()
            item.moveBy(pos.x() - event.lastScenePos().x(), pos.y() - event.lastScenePos().y())
        return item

    def draw_Item(self, event):
        """
        Draw or redraw the currently selected Item if it is movable

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        item = self.get_selected_Item()
        pos = event.scenePos()
        if item and (item.flags() & QGraphicsItem.ItemIsMovable):
            item_pos = item.scenePos()
            w, h = np.max([round_to_even(pos.x() - item_pos.x()), 5]), np.max(
                [round_to_even(pos.y() - item_pos.y()), 5])
            rect = QRect(0, 0, w, h)
            item.setRect(rect)
        elif PyDetecDiv.current_drawing_tool == DrawingTools.DrawRect:
            item = self.addRect(QRect(0, 0, 5, 5))
            item.setPen(self.pen)
            item.setPos(QPoint(pos.x(), pos.y()))
            item.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
            item.setData(0, f'Region{len(self.items())}')
            item.setZValue(10)
            item.setSelected(True)
        if item:
            self.display_Item_size(item)
        return item

    def set_Item_width(self, width):
        """
        Sets the width of the selected item

        :param width: the width value
        """
        item = self.get_selected_Item()
        if item and (item.flags() & QGraphicsItem.ItemIsMovable):
            rect = QRect(0, 0, width, item.rect().height())
            item.setRect(rect)

    def set_Item_height(self, height):
        """
        Sets the height of the selected item

        :param height: the height value
        """
        item = self.get_selected_Item()
        if item and (item.flags() & QGraphicsItem.ItemIsMovable):
            rect = QRect(0, 0, item.rect().width(), height)
            item.setRect(rect)

    def display_Item_size(self, item):
        """
        Displays the item size in the Drawing tools palette

        :param item: the item
        """
        PyDetecDiv.main_window.drawing_tools.roi_width.setValue(item.rect().width())
        PyDetecDiv.main_window.drawing_tools.roi_height.setValue(item.rect().height())


class Layer(QGraphicsItem):
    """
    A graphics item containing sub-items in order to define layer behaviour
    """

    def __init__(self, viewer, **kwargs):
        super().__init__(**kwargs)
        self.viewer = viewer

    @property
    def zIndex(self):
        """
        A convenience method returning the Z index of the Layer

        :return: the z index
        """
        return int(self.zValue())

    @zIndex.setter
    def zIndex(self, zIndex: int):
        self.setZValue(min(len(self.viewer.layers), max(1, zIndex)))

    def move_up(self):
        """
        Moves the layer up one level
        """
        self.viewer.move_layer(self.zIndex, self.zIndex + 1)

    def move_down(self):
        """
        Moves the layer down one level
        """
        self.viewer.move_layer(self.zIndex, self.zIndex - 1)

    def toggleVisibility(self):
        """
        Toggles visibility of the layer
        """
        self.setVisible(not self.isVisible())

    def addItem(self, item):
        """
        Adds an item to the layer

        :param item: the added item
        :return: the added item
        """
        item.setParentItem(self)
        return item

    def boundingRect(self):
        """
        Returns the bounding rect of the layer, which contains all children

        :return: the bounding rect of the layer
        """
        if self.childItems():
            return self.childrenBoundingRect()
        return QRectF(0.0, 0.0, 0.0, 0.0)

    def paint(self, painter, option, widget=...):
        """
        Paint method added to comply with the implementation of abstract class
        """
        pass


class BackgroundLayer(Layer):
    """
    A particular layer that is always at the bottom of a scene
    """

    @property
    def zIndex(self):
        """
        A convenience method returning the Z index of the Layer. As this is a background layer, the Z index
        should always be 0

        :return: the z index
        """
        return int(self.zValue())

    @zIndex.setter
    def zIndex(self, zIndex: int):
        zIndex = 0
        self.setZValue(zIndex)
