import qimage2ndarray
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QImage, QPen, QBitmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem
from sklearn.preprocessing import minmax_scale

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers import GraphicsView
from pydetecdiv.domain import Image, ImgDType


class ImageViewer(GraphicsView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = []
        self.background = self.addLayer(background=True)

    def setBackgroundImage(self, image_resource_data, C=0, T=0, Z=0, crop=None):
        return self.background.addImage(image_resource_data, C=C, T=T, Z=Z, crop=crop)

    def addLayer(self, background=False):
        layer = BackgroundLayer(self) if background else ImageLayer(self)
        self.scene().addItem(layer)
        layer.setZValue(len(self.layers))
        self.layers.append(layer)
        return layer

    def move_layer(self, origin, destination):
        layer = self.layers.pop(origin)
        self.layers.insert(min(len(self.layers), max(1, destination)), layer)
        for i, l in enumerate(self.layers):
            l.zIndex = i


class ImageLayer(QGraphicsItem):
    def __init__(self, viewer, **kwargs):
        super().__init__(**kwargs)
        self.T = 0
        self.viewer = viewer

    @property
    def zIndex(self):
        return int(self.zValue())

    @zIndex.setter
    def zIndex(self, zIndex: int):
        zIndex = min(len(self.viewer.layers), max(1, zIndex))
        self.setZValue(zIndex)

    def move_up(self):
        self.viewer.move_layer(self.zIndex, self.zIndex + 1)

    def move_down(self):
        self.viewer.move_layer(self.zIndex, self.zIndex - 1)

    def toggleVisibility(self):
        self.setVisible(not self.isVisible())

    def addImage(self, image_resource_data, C=0, T=0, Z=0, crop=None, transparent=None, alpha=False):
        self.T = T
        return ImageItem(image_resource_data, C=C, T=T, Z=Z, crop=crop, transparent=transparent, parent=self,
                         alpha=alpha)

    def addItem(self, item):
        item.setParentItem(self)
        return item

    def boundingRect(self):
        if self.childItems():
            return self.childrenBoundingRect()
        return QRectF(0.0, 0.0, 0.0, 0.0)

    def paint(self, painter, option, widget=...):
        pass


class BackgroundLayer(ImageLayer):

    @property
    def zIndex(self):
        return int(self.zValue())

    @zIndex.setter
    def zIndex(self, zIndex: int):
        zIndex = 0
        self.setZValue(zIndex)


class ImageItem(QGraphicsPixmapItem):
    def __init__(self, image_resource_data=None, C=0, T=0, Z=0, crop=None, transparent=None, alpha=False, parent=None):
        if image_resource_data:
            pixmap = ImageItem.get_pixmap(image_resource_data, C=C, T=T, Z=Z, crop=crop, alpha=alpha)
        else:
            pixmap = QPixmap()
        if transparent:
            pixmap.setMask(pixmap.createMaskFromColor(transparent, Qt.MaskInColor))
        super().__init__(pixmap, parent)
        self.image_resource_data = image_resource_data
        self.C = C
        self.T = T
        self.Z = Z
        self.crop = crop
        if crop is not None:
            self.setOffset(crop[0].start, crop[1].start)

    def setMask(self, mask):
        self.pixmap().setMask(QBitmap.fromPixmap(mask))

    def display(self, C=None, T=None, Z=None, alpha=False, transparent=None):
        C = self.C if C is None else C
        T = self.T if T is None else T
        Z = self.Z if Z is None else Z

        pixmap = ImageItem.get_pixmap(self.image_resource_data, C=C, T=T, Z=Z, crop=self.crop, alpha=alpha)
        if transparent:
            pixmap.setMask(pixmap.createMaskFromColor(transparent, Qt.MaskInColor))
        self.setPixmap(pixmap)

    @staticmethod
    def get_pixmap(image_resource_data, C=0, T=0, Z=0, crop=None, alpha=False):
        arr = Image.auto_channels(image_resource_data, C=C, T=T, Z=Z, crop=crop,
                                  drift=PyDetecDiv.apply_drift).as_array(np.uint8)
        if alpha:
            if len(arr.shape) == 2:
                arr = np.dstack((arr, arr, arr, arr))
            else:
                alpha_channel = np.max(arr, axis=2)
                arr = np.dstack((arr, alpha_channel))
        return QPixmap.fromImage(qimage2ndarray.array2qimage(arr), Qt.AutoColor)
