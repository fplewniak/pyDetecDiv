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
        self.backgroundImage = None

    def setBackgroundImage(self, image_resource_data, C=0, T=0, Z=0, crop=None):
        self.backgroundImage = ImageItem(image_resource_data, C=C, T=T, Z=Z, crop=crop)
        self.scene().addItem(self.backgroundImage)

    def display(self, C=None, T=None, Z=None):
        C = self.backgroundImage.C if C is None else C
        T = self.backgroundImage.T if T is None else T
        Z = self.backgroundImage.Z if Z is None else Z

        arr = Image.auto_channels(self.backgroundImage.image_resource_data, C=C, T=T, Z=Z,
                                  drift=PyDetecDiv.apply_drift).as_array(np.float64)

        if arr is not None:
            if self.backgroundImage.crop is not None:
                arr = arr[..., self.backgroundImage.crop[1], self.backgroundImage.crop[0]]

        arr *= 255.0 / arr.max()
        img = qimage2ndarray.array2qimage(arr.astype(np.uint8))
        self.backgroundImage.setPixmap(QPixmap.fromImage(img, Qt.AutoColor))

    def addLayer(self):
        layer = ImageLayer(self)
        self.scene().addItem(layer)
        return layer


class ImageLayer(QGraphicsItem):
    def __init__(self, viewer, **kwargs):
        super().__init__(**kwargs)
        self.T = 0
        self.viewer = viewer

    def toggleVisibility(self):
        self.setVisible(not self.isVisible())

    def addImage(self, image_resource_data, C=0, T=0, Z=0, crop=None, transparent=None, alpha=False):
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

    @staticmethod
    def get_pixmap(image_resource_data, C=0, T=0, Z=0, crop=None, alpha=False):
        arr = Image.auto_channels(image_resource_data, C=C, T=T, Z=Z, crop=crop,
                                  drift=PyDetecDiv.apply_drift).as_array(np.float64)
        arr = (arr * 255.0)
        if alpha:
            if len(arr.shape) == 2:
                arr = np.dstack((arr, arr, arr))
            alpha_channel = np.sum(arr, axis=2, dtype=np.float64)
            alpha_channel = 255.0 * alpha_channel / alpha_channel.max()
            arr = np.dstack((arr, alpha_channel))
        return QPixmap.fromImage(qimage2ndarray.array2qimage(arr.astype(np.uint8)), Qt.AutoColor)
