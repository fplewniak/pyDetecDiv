#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 A class defining the business logic methods that can be applied to images
"""
from __future__ import annotations

from enum import Enum

import cv2
import matplotlib.axes
import numpy as np
import tensorflow as tf
from skimage import exposure
from PIL import Image as PILimage, ImageOps

from pydetecdiv.domain import ImageResourceData, ROI


class ImgDType(Enum):
    """
    Enumeration of common names for numpy array/tensor dtypes
    """
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

    def __init__(self, array_dtype: np.dtype, tensor_dtype: tf.dtypes.DType):
        self.array_dtype = array_dtype
        self.tensor_dtype = tensor_dtype


class Image:
    """
    A business-logic class defining valid operations and attributes of 2D images
    """

    def __init__(self, data: np.ndarray | tf.Tensor = None):
        self.tensor = data if tf.is_tensor(data) else tf.convert_to_tensor(data)
        self._initial_tensor = self.tensor

    def reset(self) -> None:
        """
        Resets the tensor values to their initial state
        """
        self.tensor = self._initial_tensor

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        the shape of this Image

        :return: shape tuple
        """
        return self.tensor.shape

    @property
    def dtype(self) -> tf.dtypes.DType:
        """
        the dtype for this Image

        :return: dtype
        """
        return self.tensor.dtype

    def as_array(self, dtype: ImgDType = None, grayscale: bool = False) -> np.ndarray:
        """
        property returning the image data for this image

        :return: the image data
        :rtype: numpy.array
        """
        return self.as_tensor(dtype=dtype, grayscale=grayscale).numpy()

    def as_tensor(self, dtype: ImgDType = None, grayscale: bool = False) -> tf.Tensor:
        """
        Returns the Image as a tensor

        :param dtype: the dtype for the tensor
        :param grayscale: bool indicating whether the tensor should be 2D (grayscale) or 3D (RGB)
        :return:
        """
        tensor = self.tensor if dtype is None else self._convert_to_dtype(dtype=dtype)
        if grayscale:
            return tf.image.rgb_to_grayscale(tensor)
        return tensor

    def _convert_to_dtype(self, dtype: ImgDType = ImgDType.uint16) -> tf.Tensor:
        """
        Converts the Image to a specified dtype tensor

        :param dtype: the dtype for the tensor
        :return: the tensor of the requiested dtype
        """
        if isinstance(dtype, ImgDType):
            dtype = dtype.tensor_dtype
        saturate = (self.tensor.dtype.is_floating and dtype.is_integer) or (
                not self.tensor.dtype.is_unsigned and dtype.is_unsigned)
        return tf.image.convert_image_dtype(self.tensor, dtype=dtype, saturate=saturate)

    def rgb_to_gray(self) -> Image:
        """
        Return a grayscale Image obtained from an RGB Image

        :return: grayscale Image
        """
        return Image(self._rgb_to_gray())

    def _rgb_to_gray(self) -> tf.Tensor:
        """
        Return a grayscale 2D tensor obtained from an RGB 3D tensor

        :return: 2D tensor
        """
        return tf.image.rgb_to_grayscale(self.tensor)

    def warp_affine(self, affine_matrix: np.ndarray, in_place: bool = True) -> Image:
        tensor = tf.convert_to_tensor(cv2.warpAffine(self.as_array(), np.float32(affine_matrix), (self.shape[1], self.shape[0])))
        if in_place is False:
            return Image(tensor)
        self.tensor = tensor
        self.tensor = self._convert_to_dtype(dtype=self._initial_tensor.dtype)
        return self

    def resize(self, shape: tuple[int, int] = None, method: str = 'nearest', antialias: bool = True) -> Image:
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

    def show(self, ax: matplotlib.axes.Axes, grayscale: bool = False, **kwargs):
        """
        Show the Image as a matplotlib image plot

        :param ax: the matplotlib ax to plot the image in
        :param grayscale: bool whether defining whether the image should be displayed as grayscale
        :param kwargs: keyword arguments passed to the imshow method
        """
        ax.imshow(self.as_array(ImgDType.uint8, grayscale), **kwargs)

    def histogram(self, ax: matplotlib.axes.Axes, bins: str = 'auto', color: str = 'black'):
        """
        Display a histogram of values

        :param ax: the matplotlib ax toplot the histogram in
        :param bins: the number of bins
        :param color: the color
        """
        ax.hist(self.as_array().flatten(), bins=bins, histtype='step', color=color)

    def channel_histogram(self, ax: matplotlib.axes.Axes, bins: str = 'auto', ):
        """
        Returns a histogram of channels' values
        :param ax: the matplotlib ax to plot the histogram in
        :param bins: the number of bins
        """
        colours = ['red', 'green', 'blue', 'yellow']
        if len(self.shape) != 2:
            ax.hist(self._rgb_to_gray().numpy().flatten(), bins=bins, histtype='step', color='black')
            for c in range(self.tensor.shape[-1]):
                ax.hist(self.as_array()[..., c].flatten(), bins=bins, histtype='step', color=colours[c])
        else:
            self.histogram(ax, bins=bins)

    def crop(self, offset_height: int, offset_width: int, target_height: int, target_width: int, new_image: bool = True) -> Image:
        """
        Crop the current Image

        :param offset_height: the Y offset
        :param offset_width: the X offset
        :param target_height: the height of the cropped image
        :param target_width: the width of the cropped image
        :param new_image: if True, returns a new Image, otherwise, the current Image is replaced with its cropped
        version
        :return: the cropped Image
        """
        tensor = tf.expand_dims(self.tensor, axis=-1) if len(self.shape) == 2 else self.tensor
        tensor = tf.squeeze(
                tf.image.crop_to_bounding_box(tensor, offset_height, offset_width, target_height, target_width))
        if new_image:
            return Image(tensor)
        self.tensor = tensor
        return self

    def auto_contrast(self, preserve_tone: bool = True) -> Image:
        """
        Adjust contrast automatically using PIL package. RGBA images cannot be used here
        :param preserve_tone: if True, the tone is preserved
        :return: the current Image after correction
        """
        self.tensor = tf.convert_to_tensor(
                np.array(ImageOps.autocontrast(PILimage.fromarray(self.as_array(np.uint8)), preserve_tone=preserve_tone)))
        return self

    def adjust_contrast(self, factor: float = 2.0) -> Image:
        """
        Automatic contrast adjustment

        :param factor: the contrast adjustment factor
        :return: the current Image after correction
        """
        self.tensor = tf.image.adjust_contrast(self.tensor, factor)
        return self

    def stretch_contrast(self, q: tuple[int, int] = None) -> Image:
        """
        Stretches the contrast of the Image

        :param q: the quantile values for correction, the qlow will be set to 0 and the qhigh to 1
        :return: the current Image after correction
        """
        img = self.as_array()
        if np.max(img) > np.min(img):
            if q is None:
                q = [0.001, 0.999]
            qlow, qhi = np.quantile(img[img > 0.0], q)
            self.tensor = tf.convert_to_tensor(exposure.rescale_intensity(img, in_range=(qlow, qhi)))
        return self

    def equalize_hist(self, adapt: bool = False) -> Image:
        """
        Adjust exposure using the histogram equalization method

        :param adapt: bool to set adaptative method
        :return: the current Image after correction
        """
        if adapt:
            self.tensor = tf.convert_to_tensor(exposure.equalize_adapthist(self.as_array(ImgDType.float64)))
        else:
            self.tensor = tf.convert_to_tensor(exposure.equalize_hist(self.as_array(ImgDType.float64)))
        self.tensor = self._convert_to_dtype(dtype=self._initial_tensor.dtype)
        return self

    def sigmoid_correction(self) -> Image:
        """
        Exposure correction using the sigmoid method

        :return: the current Image after correction
        """
        self.tensor = tf.convert_to_tensor(exposure.adjust_sigmoid(self.as_array()))
        return self

    def decompose_channels(self) -> list[Image]:
        """
        Split an RGB image in a list of channels

        :return: list of one Image per channel
        """
        if len(self.shape) == 2:
            return [self]
        return [Image(array) for array in tf.unstack(self.tensor, axis=-1)]

    @staticmethod
    def add(images: list[Image]) -> Image:
        """
        Pixelwise addition of images in a list

        :param images: the list of images
        :return: the resulting Image
        """
        return Image(tf.math.add_n([i.tensor for i in images]))

    @staticmethod
    def mean(images: list[Image]) -> Image:
        """
        Compute the pixelwise mean of a list of images

        :param images: the list of images to average
        :return: the averaged image
        """
        if len(images) > 1:
            tensor = tf.math.add_n([i.as_tensor(ImgDType.float32) / len(images) for i in images])
            return Image(tf.image.convert_image_dtype(tensor, images[0].tensor.dtype))
        return images[0]

    @staticmethod
    def compose_channels(channels: list[Image] | tuple[Image], alpha: bool = False) -> Image:
        """
        Compose 3 channels into an RGB image, optionally adding an alpha channel determined as the maximum of the three
        channels if alpha is True

        :param channels: the three channels to compose
        :param alpha: bool, True if alpha should be added
        :return:
        """
        if alpha:
            channels.append(Image(tf.math.maximum(tf.math.maximum(channels[0].as_tensor(), channels[1].as_tensor()),
                                                  channels[2].as_tensor())))
        return Image(tf.stack([c.as_tensor() for c in channels], axis=-1))

    @staticmethod
    def auto_channels(image_resource_data: ImageResourceData, C: int = 0, T: int = 0, Z: int | list[int] | tuple[int] = 0,
                      crop: tuple[slice, slice] = None, drift: bool = False, alpha: bool = False) -> Image:
        """
        Returns a RGB, RGBA or grayscale image depending upon the C or Z values. If C (or Z) is a tuple, it is used as
        RGB values. If alpha is set to True, then the maximum value of every pixel across all channels defines its
        alpha value. If C and Z are both an index, then the returned image is grayscale.

        :param image_resource_data: the image resource data used to create the Image
        :param C: the channel or channels tuple
        :param T: the time frame index
        :param Z: the z-slice or z-slices tuple
        :param crop: a tuple defining the crop values as slices = (slice(xmin, xmax), slice(ymin, ymax))
        :param drift: bool defining whether drift correction should be applied
        :param alpha: bool defining whether the image should contain an alpha channel
        :return: Image
        """
        img = None
        if crop is None:
            crop = (None, None)
        if isinstance(C, int):
            if isinstance(Z, (tuple, list)):
                img = Image.compose_channels(
                        [Image(image_resource_data.image(C=C, T=T, Z=c, sliceX=crop[0], sliceY=crop[1], drift=drift)) for c
                         in Z], alpha=alpha)
            else:
                img = Image(image_resource_data.image(C=C, T=T, Z=Z, sliceX=crop[0], sliceY=crop[1], drift=drift))
        elif isinstance(C, (tuple, list)):
            img = Image.compose_channels(
                    [Image(image_resource_data.image(C=c, T=T, Z=Z, sliceX=crop[0], sliceY=crop[1], drift=drift)) for c in
                     C], alpha=alpha)
        return img


def get_images_sequences(imgdata: ImageResourceData, roi_list: list[ROI], t: int, seqlen: int = None, z: list[int, int, int] = None,
                         apply_drift: bool = True) -> tf.Tensor:
    """
    Get a sequence of seqlen images for each roi

    :param imgdata: the image data resource
    :param roi_list: the list of ROIs
    :param t: the starting time point (index of frame)
    :param seqlen: the number of frames
    :param z: the z-layers to stack
    :param apply_drift: True if drift must be applied, False otherwise
    :return: a tensor containing the sequences for all ROIs
    """
    maxt = min(imgdata.sizeT, t + seqlen) if seqlen else imgdata.sizeT
    roi_sequences = tf.stack([get_rgb_images_from_stacks(imgdata, roi_list, f, z=z) for f in range(t, maxt)], axis=1,
                             apply_drift=apply_drift)
    if roi_sequences.shape[1] < seqlen:
        padding_config = [[0, 0], [seqlen - roi_sequences.shape[1], 0], [0, 0], [0, 0], [0, 0]]
        roi_sequences = tf.pad(roi_sequences, padding_config, mode='CONSTANT', constant_values=0.0)
    # print('roi sequence', roi_sequences.shape)
    return roi_sequences


def get_rgb_images_from_stacks_memmap(imgdata: ImageResourceData, roi_list: list[ROI], t: int, z: list[int, int, int] = None,
                                      apply_drift: bool = True) -> list[tf.Tensor]:
    """
    Combine 3 z-layers of a grayscale image resource into a RGB image where each of the z-layer is a channel

    :param imgdata: the image data resource
    :param roi_list: the list of ROIs
    :param t: the frame index
    :param z: a list of 3 z-layer indices defining the grayscale layers that must be combined as channels
    :param apply_drift: True if drift must be applied, False otherwise
    :return: a tensor of the combined RGB images
    """
    if z is None:
        z = [0, 0, 0]
    roi_images = [
        Image.compose_channels([Image(imgdata.image_memmap(sliceX=slice(roi.x, roi.x + roi.width),
                                                           sliceY=slice(roi.y, roi.y + roi.height),
                                                           C=0, Z=z[0], T=t,
                                                           drift=apply_drift)).stretch_contrast(),
                                Image(imgdata.image_memmap(sliceX=slice(roi.x, roi.x + roi.width),
                                                           sliceY=slice(roi.y, roi.y + roi.height),
                                                           C=0, Z=z[1], T=t,
                                                           drift=apply_drift)).stretch_contrast(),
                                Image(imgdata.image_memmap(sliceX=slice(roi.x, roi.x + roi.width),
                                                           sliceY=slice(roi.y, roi.y + roi.height),
                                                           C=0, Z=z[2], T=t,
                                                           drift=apply_drift)).stretch_contrast(),
                                ]).as_tensor(ImgDType.float32) for roi in roi_list]
    return roi_images


def stack_fov_image(imgdata: ImageResourceData, t: int, z: list[int, int, int] = None, apply_drift: bool = True) -> tf.Tensor:
    """
    Combine 3 z-layers of a grayscale full image resource (one complete FOV) into a RGB image where each of the z-layer is a channel

    :param imgdata: the image data resource
    :param t: the frame index
    :param z: a list of 3 z-layer indices defining the grayscale layers that must be combined as channels
    :param apply_drift: True if drift must be applied, False otherwise
    :return: a tensor of the combined RGB images
    """
    if z is None:
        z = [0, 0, 0]

    image1 = Image(imgdata.image(T=t, Z=z[0], drift=apply_drift))
    image2 = Image(imgdata.image(T=t, Z=z[1], drift=apply_drift))
    image3 = Image(imgdata.image(T=t, Z=z[2], drift=apply_drift))

    rgb_image = Image.compose_channels([image1.stretch_contrast(),
                                        image2.stretch_contrast(),
                                        image3.stretch_contrast()
                                        ]).as_tensor(ImgDType.float32)
    return rgb_image


def get_rgb_images_from_stacks(imgdata: ImageResourceData, roi_list: list[ROI], t: int, z: list[int, int, int] = None,
                               apply_drift: bool = True) -> list[tf.Tensor]:
    """
    Combine 3 z-layers of a grayscale image resource into a RGB image where each of the z-layer is a channel

    :param imgdata: the image data resource
    :param roi_list: the list of ROIs
    :param t: the frame index
    :param z: a list of 3 z-layer indices defining the grayscale layers that must be combined as channels
    :param apply_drift: True if drift must be applied, False otherwise
    :return: a tensor of the combined RGB images
    """
    if z is None:
        z = [0, 0, 0]

    image1 = Image(imgdata.image(T=t, Z=z[0], drift=apply_drift))
    image2 = Image(imgdata.image(T=t, Z=z[1], drift=apply_drift))
    image3 = Image(imgdata.image(T=t, Z=z[2], drift=apply_drift))

    roi_images = [Image.compose_channels([image1.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast(),
                                          image2.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast(),
                                          image3.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast()
                                          ]).as_tensor(ImgDType.float32) for roi in roi_list]
    return roi_images
