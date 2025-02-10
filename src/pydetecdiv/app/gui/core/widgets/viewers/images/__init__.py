"""
A module defining classes to view Images in a layered viewer
"""
import qimage2ndarray
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QBitmap, QColor
from PySide6.QtWidgets import QGraphicsPixmapItem, QWidget

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers import GraphicsView, Layer, BackgroundLayer
from pydetecdiv.domain.Image import Image, ImgDType
from pydetecdiv.domain.ImageResourceData import ImageResourceData


class ImageViewer(GraphicsView):
    """
    A class defining an Image viewer
    """

    def __init__(self, parent: QWidget = None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.image_resource_data = None

    def _create_layer(self, background: bool = False) -> 'ImageLayer':
        """
        Create a new layer

        :param background: if True, the created layer is a BackgroundLayer
        :return: the created layer
        """
        if background:
            return BackgroundImageLayer(self)
        return ImageLayer(self)

    def setBackgroundImage(self, image_resource_data: ImageResourceData, C: int = 0, T: int = 0, Z: int = 0,
                           crop: tuple[slice, slice] = None) -> 'ImageItem':
        """
        Sets the background image

        :param image_resource_data: the image resource data
        :param C: the channel or tuple of channels
        :param T: the frame index
        :param Z: the Z-slice
        :param crop: the crop values
        :return: the background image
        """
        self.image_resource_data = image_resource_data
        return self.background.setImage(image_resource_data, C=C, T=T, Z=Z, crop=crop)

    def display(self, T: int = None):
        """
        Display the viewer at a given time frame

        :param T: the time frame index
        """
        for layer in self.layers:
            if layer.image:
                layer.image.display(T=T)


class ImageLayer(Layer):
    """
    A class defining an Image layer
    """

    def __init__(self, viewer: ImageViewer, **kwargs):
        super().__init__(viewer, **kwargs)
        self.T = 0
        self.image = None

    def setImage(self, image_resource_data: ImageResourceData, C: int = 0, T: int = 0, Z: int = 0, crop: tuple[slice, slice] = None,
                 transparent: QColor = None, alpha: bool = False):
        """
        Adds an image to the current layer

        :param image_resource_data: the image resource data
        :param C: the channel or tuple of channels
        :param T: the time frame index
        :param Z: the Z-slice
        :param crop: the crop values
        :param transparent: a transparency mask (where black is transparent)
        :param alpha: if True, the image is set to RGBA
        :return: the image item
        """
        self.T = T
        if self.image is not None:
            self.scene().removeItem(self.image)
        self.image = ImageItem(image_resource_data, C=C, T=T, Z=Z, crop=crop, transparent=transparent, parent=self,
                               alpha=alpha)
        return self.image


class BackgroundImageLayer(BackgroundLayer, ImageLayer):
    """
    A class defining a background image
    """

    def __init__(self, viewer: ImageViewer, **kwargs):
        super().__init__(viewer, **kwargs)


class ImageItem(QGraphicsPixmapItem):
    """
    A class defining an ImageItem
    """

    def __init__(self, image_resource_data: ImageResourceData = None, C=0, T=0, Z=0, crop: tuple[slice, slice] = None,
                 transparent=None, alpha=False, parent=None):
        if image_resource_data:
            pixmap = ImageItem.get_pixmap(image_resource_data, C=C, T=T, Z=Z, crop=crop, alpha=alpha)
        else:
            pixmap = QPixmap()
        if transparent:
            pixmap.setMask(pixmap.createMaskFromColor(transparent, Qt.MaskMode.MaskInColor))
        super().__init__(pixmap, parent=parent)
        self.image_resource_data = image_resource_data
        self.C = C
        self.T = T
        self.Z = Z
        self.alpha = alpha
        self.crop = crop
        # if crop is not None:
        #     self.setOffset(crop[0].start, crop[1].start)

    def get_CTZ(self):
        """
        Gets the channel, current time frame and Z-slice of the current ImageItem

        :return: a tiple with the values of C, T, Z
        """
        return (self.C, self.T, self.Z)

    def setMask(self, mask):
        """
        Sets the transparency mask

        :param mask: the transparency mask (black is transparent)
        """
        self.pixmap().setMask(QBitmap.fromPixmap(mask))

    def display(self, C=None, T=None, Z=None, transparent=None):
        """
        Displays the ImageItem

        :param C: the channel or tuple of channels
        :param T: the time frame
        :param Z: the z-slice
        :param transparent: the transparency mask
        """
        self.C = self.C if C is None else C
        self.T = self.T if T is None else T
        self.Z = self.Z if Z is None else Z

        pixmap = ImageItem.get_pixmap(self.image_resource_data, C=self.C, T=self.T, Z=self.Z, crop=self.crop,
                                      alpha=self.alpha)
        if transparent:
            pixmap.setMask(pixmap.createMaskFromColor(transparent, Qt.MaskMode.MaskInColor))
        self.setPixmap(pixmap)

    def set_channel(self, C):
        """
        Sets the current channel

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
    def get_pixmap(image_resource_data, C=0, T=0, Z=0, crop: tuple[slice, slice] = None, alpha=False):
        """
        Gets a pixmap from image resource data

        :param image_resource_data: image resource data
        :param C: the channel or tuple of channels
        :param T: the time frame
        :param Z: the z-slice
        :param crop: the crop values
        :param alpha: True to request RGBA
        :return: a QPixmap object
        """
        arr = Image.auto_channels(image_resource_data, C=C, T=T, Z=Z, crop=crop,
                                  drift=PyDetecDiv.apply_drift, alpha=alpha).stretch_contrast().channel_last(ImgDType.uint8).numpy()
        # if alpha:
        #     if len(arr.shape) == 2:
        #         arr = np.dstack((arr, arr, arr, arr))
        #     else:
        #         alpha_channel = np.max(arr, axis=2)
        #         arr = np.dstack((arr, alpha_channel))
        return QPixmap.fromImage(qimage2ndarray.array2qimage(arr), Qt.ImageConversionFlag.AutoColor)
