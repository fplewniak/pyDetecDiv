import qimage2ndarray
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QImage, QPen, QBitmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem
from sklearn.preprocessing import minmax_scale

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers import GraphicsView, Layer, BackgroundLayer
from pydetecdiv.domain import Image, ImgDType


class ImageViewer(GraphicsView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_resource_data = None
        self.scale_value = 100

    def _create_layer(self, background=False):
        if background:
            return BackgroundImageLayer(self)
        return ImageLayer(self)

    def setBackgroundImage(self, image_resource_data, C=0, T=0, Z=0, crop=None):
        self.image_resource_data = image_resource_data
        return self.background.addImage(image_resource_data, C=C, T=T, Z=Z, crop=crop)

    def display(self, T=None):
        for layer in self.layers:
            if layer.image:
                layer.image.display(T=T)

    def zoom_set_value(self, value):
        self.scale(value / self.scale_value, value / self.scale_value)
        self.scale_value = value

    def zoom_fit(self):
        """
        Set the zoom value to fit the image in the viewer
        """
        # self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        self.scale_value = int(100 * np.around(self.transform().m11(), 2))


class ImageLayer(Layer):
    def __init__(self, viewer, **kwargs):
        super().__init__(viewer, **kwargs)
        self.T = 0
        self.image = None

    def addImage(self, image_resource_data, C=0, T=0, Z=0, crop=None, transparent=None, alpha=False):
        self.T = T
        self.image = ImageItem(image_resource_data, C=C, T=T, Z=Z, crop=crop, transparent=transparent, parent=self,
                               alpha=alpha)
        return self.image


class BackgroundImageLayer(BackgroundLayer, ImageLayer):
    def __init__(self, viewer, **kwargs):
        super().__init__(viewer, **kwargs)


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
        self.alpha = alpha
        self.crop = crop
        if crop is not None:
            self.setOffset(crop[0].start, crop[1].start)

    def get_CTZ(self):
        return (self.C, self.T, self.Z)

    def setMask(self, mask):
        self.pixmap().setMask(QBitmap.fromPixmap(mask))

    def display(self, C=None, T=None, Z=None, transparent=None):
        self.C = self.C if C is None else C
        self.T = self.T if T is None else T
        self.Z = self.Z if Z is None else Z

        pixmap = ImageItem.get_pixmap(self.image_resource_data, C=self.C, T=self.T, Z=self.Z, crop=self.crop,
                                      alpha=self.alpha)
        if transparent:
            pixmap.setMask(pixmap.createMaskFromColor(transparent, Qt.MaskInColor))
        self.setPixmap(pixmap)

    def set_channel(self, C):
        """
        Sets the current channel
        TODO: allow specification of channel by name, this method should set the self.C field to the index corresponding
        TODO: to the requested name if the C argument is a str

        :param C: index of the current channel
        :type C: int or tuple(int, int, int) for RGB or tuple(int, int, int, int) for RGBA
        """
        self.C = C

    def set_Z(self, Z=0):
        """
        Set the layer to the specified value and refresh the display

        :param Z: the Z layer index
        :type Z: int
        """
        if self.Z != Z:
            self.Z = Z
            self.display()

    def change_frame(self, T=0):
        """
        Change the current frame to the specified time index and refresh the display

        :param T:
        """
        if self.T != T:
            self.T = T

    @staticmethod
    def get_pixmap(image_resource_data, C=0, T=0, Z=0, crop=None, alpha=False):
        arr = Image.auto_channels(image_resource_data, C=C, T=T, Z=Z, crop=crop,
                                  drift=PyDetecDiv.apply_drift, alpha=alpha).as_array(np.uint8)
        # if alpha:
        #     if len(arr.shape) == 2:
        #         arr = np.dstack((arr, arr, arr, arr))
        #     else:
        #         alpha_channel = np.max(arr, axis=2)
        #         arr = np.dstack((arr, alpha_channel))
        return QPixmap.fromImage(qimage2ndarray.array2qimage(arr), Qt.AutoColor)
