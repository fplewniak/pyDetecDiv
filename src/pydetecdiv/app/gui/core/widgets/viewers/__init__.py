import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, QRectF
from PySide6.QtGui import QKeySequence, QTransform, QPen
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem

from pydetecdiv.app import PyDetecDiv, DrawingTools
from pydetecdiv.utils import round_to_even


class GraphicsView(QGraphicsView):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setScene(Scene())
        self.layers = []
        self.background = self.addLayer(background=True)
        # sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        # self.setSizePolicy(sizePolicy)

    def _create_layer(self, background=False):
        if background:
            return BackgroundLayer(self)
        return Layer(self)

    def addLayer(self, background=False):
        layer = self._create_layer(background)
        self.scene().addItem(layer)
        layer.setZValue(len(self.layers))
        self.layers.append(layer)
        return layer

    def move_layer(self, origin, destination):
        layer = self.layers.pop(origin)
        self.layers.insert(min(len(self.layers), max(1, destination)), layer)
        for i, l in enumerate(self.layers):
            l.zIndex = i

class Scene(QGraphicsScene):
    def __init__(self, **kwargs):
        super().__init__()
        self.pen = QPen(Qt.GlobalColor.cyan, 2)
        self.match_pen = QPen(Qt.GlobalColor.yellow, 2)
        self.saved_pen = QPen(Qt.GlobalColor.green, 2)
        self.warning_pen = QPen(Qt.GlobalColor.red, 2)

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
                case DrawingTools.DrawROI:
                    self.select_Item(event)
                case DrawingTools.DuplicateROI:
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
                case DrawingTools.DrawROI, Qt.NoModifier:
                    self.draw_Item(event)
                case DrawingTools.DrawROI, Qt.ControlModifier:
                    self.move_Item(event)
                case DrawingTools.DuplicateROI, Qt.NoModifier:
                    self.move_Item(event)

    def select_Item(self, event):
        """
        Select the current area/Item

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        _ = [r.setSelected(False) for r in self.items()]
        r = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
        if not isinstance(r, Layer):
            r.setSelected(True)
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

            # if [r for r in item.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
            #     item.setPen(self.warning_pen)
            # else:
            #     item.setPen(self.pen)

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
            # if [r for r in item.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
            #     item.setPen(self.warning_pen)
            # else:
            #     item.setPen(self.pen)

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
        else:
            item = self.addRect(QRect(0, 0, 5, 5))
            item.setPen(self.pen)
            item.setPos(QPoint(pos.x(), pos.y()))
            item.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
            item.setData(0, f'Region{len(self.items())}')
            item.setZValue(10)
            # self.select_Item(event)
            item.setSelected(True)

        # if [r for r in item.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
        #     item.setPen(self.warning_pen)
        # else:
        #     item.setPen(self.pen)
        self.display_Item_size(item)

    def set_Item_width(self, width):
        item = self.get_selected_Item()
        if item and (item.flags() & QGraphicsItem.ItemIsMovable):
            rect = QRect(0, 0, width, item.rect().height())
            item.setRect(rect)

    def set_ROI_width(self, width):
        self.set_Item_width(width)

    def set_Item_height(self, height):
        item = self.get_selected_Item()
        if item and (item.flags() & QGraphicsItem.ItemIsMovable):
            rect = QRect(0, 0, item.rect().width(), height)
            item.setRect(rect)

    def set_ROI_height(self, height):
        self.set_Item_height(height)

    def display_Item_size(self, item):
        PyDetecDiv.main_window.drawing_tools.roi_width.setValue(item.rect().width())
        PyDetecDiv.main_window.drawing_tools.roi_height.setValue(item.rect().height())


class Layer(QGraphicsItem):
    def __init__(self, viewer, **kwargs):
        super().__init__(**kwargs)
        self.viewer = viewer

    @property
    def zIndex(self):
        return int(self.zValue())

    @zIndex.setter
    def zIndex(self, zIndex: int):
        self.setZValue(min(len(self.viewer.layers), max(1, zIndex)))

    def move_up(self):
        self.viewer.move_layer(self.zIndex, self.zIndex + 1)

    def move_down(self):
        self.viewer.move_layer(self.zIndex, self.zIndex - 1)

    def toggleVisibility(self):
        self.setVisible(not self.isVisible())

    def addItem(self, item):
        item.setParentItem(self)
        return item

    def boundingRect(self):
        if self.childItems():
            return self.childrenBoundingRect()
        return QRectF(0.0, 0.0, 0.0, 0.0)

    def paint(self, painter, option, widget=...):
        pass


class BackgroundLayer(Layer):

    @property
    def zIndex(self):
        return int(self.zValue())

    @zIndex.setter
    def zIndex(self, zIndex: int):
        zIndex = 0
        self.setZValue(zIndex)
