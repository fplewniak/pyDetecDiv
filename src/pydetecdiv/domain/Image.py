#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to images
"""
from enum import Enum

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import exposure


class ImgDType(Enum):
    uint8 = (np.uint8, tf.uint8)
    uint16 = (np.uint16, tf.uint16)
    uint32 = (np.uint32, tf.uint32)
    uint64 = (np.uint64, tf.uint64)
    int8 = (np.int8, tf.int8)
    int16 = (np.int16, tf.int16)
    int32 = (np.int32, tf.int32)
    int64 = (np.int64, tf.int64)
    float16 = (np.float16, tf.float16)
    float32 = (np.float32, tf.float32)
    float64 = (np.float64, tf.float64)

    def __init__(self, array_dtype, tensor_dtype):
        self.array_dtype = array_dtype
        self.tensor_dtype = tensor_dtype


class Image():
    """
    A business-logic class defining valid operations and attributes of 2D images
    """

    def __init__(self, data=None, **kwargs):
        self.tensor = data if tf.is_tensor(data) else tf.convert_to_tensor(data)
        self._initial_tensor = self.tensor

    def reset(self):
        self.tensor = self._initial_tensor

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype

    def as_array(self, dtype=None, grayscale=False):
        """
        property returning the image data for this image
        :return: the image data
        :rtype: numpy.array
        """
        return self.as_tensor(dtype=dtype, grayscale=grayscale).numpy()

    def as_tensor(self, dtype=None, grayscale=False):
        tensor = self.tensor if dtype is None else self._convert_to_dtype(dtype=dtype)
        if grayscale:
            return tf.image.rgb_to_grayscale(tensor)
        return tensor

    def _convert_to_dtype(self, dtype=ImgDType.uint16):
        if isinstance(dtype, ImgDType):
            dtype = dtype.tensor_dtype
        saturate = True if (self.tensor.dtype.is_floating and dtype.is_integer) or (
                not self.tensor.dtype.is_unsigned and dtype.is_unsigned) else False
        return tf.image.convert_image_dtype(self.tensor, dtype=dtype, saturate=saturate)

    def rgb_to_gray(self):
        return Image(self._rgb_to_gray())

    def _rgb_to_gray(self):
        return tf.image.rgb_to_grayscale(self.tensor)

    def resize(self, shape=None, method='nearest', antialias=True):
        """
        Resize image to the defined shape with the defined method.

        :param shape: the target shape
        :param method: the resizing method
            * bilinear: Bilinear interpolation. If antialias is true, becomes a hat/tent filter function with radius 1
              when downsampling.
            * lanczos3: Lanczos kernel with radius 3. High-quality practical filter but may have some ringing,
              especially on synthetic images.
            * lanczos5: Lanczos kernel with radius 5. Very-high-quality filter but may have stronger ringing.
            * bicubic: Cubic interpolant of Keys. Equivalent to Catmull-Rom kernel. Reasonably good quality and faster
              than Lanczos3Kernel, particularly when upsampling.
            * gaussian: Gaussian kernel with radius 3, sigma = 1.5 / 3.0.
            * nearest: (default) Nearest neighbour interpolation. antialias has no effect when used with nearest
              neighbour interpolation.
            * area: Anti-aliased resampling with area interpolation. antialias has no effect when used with area
              interpolation; it always anti-aliases.
            * mitchellcubic: Mitchell-Netravali Cubic non-interpolating filter. For synthetic images (especially those
              lacking proper prefiltering), less ringing than Keys cubic kernel but less sharp.
        :return: the resized Image object
        """
        tensor = tf.expand_dims(self.tensor, axis=-1) if len(self.shape) == 2 else self.tensor
        return Image(tf.squeeze(tf.image.resize(tensor, shape, method=method)))

    def show(self, ax, grayscale=False, **kwargs):
        ax.imshow(self.as_array(ImgDType.uint8, grayscale), **kwargs)

    def histogram(self, ax, bins='auto', color='black'):
        ax.hist(self.as_array().flatten(), bins=bins, histtype='step', color=color)

    def channel_histogram(self, ax, bins='auto', ):
        colours = ['red', 'green', 'blue', 'yellow']
        if len(self.shape) != 2:
            ax.hist(self._rgb_to_gray().numpy().flatten(), bins=bins, histtype='step', color='black')
            for c in range(self.tensor.shape[-1]):
                ax.hist(self.as_array()[..., c].flatten(), bins=bins, histtype='step', color=colours[c])
        else:
            self.histogram(ax, bins=bins)

    def crop(self, offset_height, offset_width, target_height, target_width, new_image=True):
        tensor = tf.expand_dims(self.tensor, axis=-1) if len(self.shape) == 2 else self.tensor
        tensor = tf.squeeze(
            tf.image.crop_to_bounding_box(tensor, offset_height, offset_width, target_height, target_width))
        if new_image:
            return Image(tensor)
        self.tensor = tensor
        return self

    def adjust_contrast(self, factor=2.0):
        self.tensor = tf.image.adjust_contrast(self.tensor, factor)
        return self

    def stretch_contrast(self, q=[0.001, 0.999]):
        img = self.as_array()
        qlow, qhi = np.quantile(img[img > 0.0], q)
        self.tensor = tf.convert_to_tensor(exposure.rescale_intensity(img, in_range=(qlow, qhi)))
        return self

    def equalize_hist(self, adapt=False):
        if adapt:
            self.tensor = tf.convert_to_tensor(exposure.equalize_adapthist(self.as_array(ImgDType.float64)))
        else:
            self.tensor = tf.convert_to_tensor(exposure.equalize_hist(self.as_array(ImgDType.float64)))
        self.tensor = self._convert_to_dtype(dtype=self._initial_tensor.dtype)
        return self

    def sigmoid_correction(self):
        self.tensor = tf.convert_to_tensor(exposure.adjust_sigmoid(self.as_array()))
        return self

    def decompose_channels(self):
        if len(self.shape) == 2:
            return [self]
        return [Image(array) for array in tf.unstack(self.tensor, axis=-1)]

    @staticmethod
    def add(images):
        return Image(tf.math.add_n([i.tensor for i in images]))

    @staticmethod
    def mean(images):
        if len(images) > 1:
            tensor = tf.math.add_n([i.as_tensor(ImgDType.float32) / len(images) for i in images])
            return Image(tf.image.convert_image_dtype(tensor, images[0].tensor.dtype))
        return images[0]

    @staticmethod
    def compose_channels(channels):
        return Image(tf.stack([c.as_tensor() for c in channels], axis=-1))
