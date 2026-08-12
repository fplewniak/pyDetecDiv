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
import torch
import torchvision as tv
import torchvision.transforms.v2
from torchvision.transforms import v2, InterpolationMode
from torchvision import tv_tensors
from skimage import exposure


class ImgDType(Enum):
    """
    Enumeration of common names for numpy array/tensor dtypes
    """
    uint8 = (np.uint8, torch.uint8)
    uint16 = (np.uint16, torch.uint16)
    uint32 = (np.uint32, torch.uint32)
    uint64 = (np.uint64, torch.uint64)
    int8 = (np.int8, torch.int8)
    int16 = (np.int16, torch.int16)
    int32 = (np.int32, torch.int32)
    int64 = (np.int64, torch.int64)
    float16 = (np.float16, torch.float16)
    float32 = (np.float32, torch.float32)
    float64 = (np.float64, torch.float64)

    def __init__(self, array_dtype: np.dtype, torch_dtype: torch.dtype):
        self.array_dtype = array_dtype
        self.torch_dtype = torch_dtype

    @staticmethod
    def get_dtype(torch_dtype: torch.dtype) -> ImgDType:
        """
        Converts a torch dtype into the corresponding ImgDType

        :param torch_dtype: the torch dtype to convert
        :return: the ImgDType for the requested dtype
        """
        types = {
            torch.uint8  : ImgDType.uint8,
            torch.uint16 : ImgDType.uint16,
            torch.uint32 : ImgDType.uint32,
            torch.uint64 : ImgDType.uint64,
            torch.int8   : ImgDType.int8,
            torch.int16  : ImgDType.int16,
            torch.int32  : ImgDType.int32,
            torch.int64  : ImgDType.int64,
            torch.float16: ImgDType.float16,
            torch.float32: ImgDType.float32,
            torch.float64: ImgDType.float64,
            }
        return types[torch_dtype]


class Image:
    """
    A business-logic class defining valid operations and attributes of 2D images
    """

    def __init__(self, data: np.ndarray | torch.Tensor):
        self.torch = data if torch.is_tensor(data) else torch.from_numpy(data.copy())
        self._initial_torch = self.torch

    def reset(self) -> None:
        """
        Resets the tensor values to their initial state
        """
        self.torch = self._initial_torch

    @property
    def shape(self) -> tuple:
        """
        the shape of this Image
        """
        return tuple(self.torch.size())

    @property
    def dtype(self) -> ImgDType:
        """
        the dtype for this Image
        """
        return ImgDType.get_dtype(self.torch.dtype)

    def as_array(self, dtype: ImgDType | None = None, grayscale: bool = False, channel_last: bool = False) -> np.ndarray:
        """
        property returning the image data for this image
        """
        np_array = self.as_torch(dtype=dtype, grayscale=grayscale).numpy()
        return np_array

    def as_tensor(self, dtype: ImgDType | None = None, grayscale: bool = False) -> torch.Tensor:
        """
        Returns the Image as a tensor

        :param dtype: the dtype for the tensor
        :param grayscale: bool indicating whether the tensor should be 2D (grayscale) or 3D (RGB)
        :return: the image as a tensor
        """
        return self.as_torch(dtype=dtype, grayscale=grayscale)

    def as_torch(self, dtype: None | ImgDType = None, grayscale: bool = False) -> torch.Tensor:
        """
        Returns the Image as a tensor

        :param dtype: the dtype for the tensor
        :param grayscale: bool indicating whether the tensor should be 2D (grayscale) or 3D (RGB)
        :return: the image as a Torch tensor
        """
        # tensor = self.torch if dtype is None else v2.ToDtype(dtype=dtype.torch_dtype, scale=True)(self.torch)
        tensor = self.torch if dtype is None else self._convert_to_dtype(dtype=dtype)
        if grayscale:
            return torch.squeeze(v2.Grayscale()(tensor))
        return tensor

    def _convert_to_dtype(self, dtype: ImgDType = ImgDType.uint16) -> torch.Tensor:
        """
        Converts the Image to a specified dtype tensor

        :param dtype: the dtype for the tensor
        :return: the tensor of the requested dtype
        """
        dtype = dtype.torch_dtype
        scale = True
        return v2.ToDtype(dtype=dtype, scale=scale)(self.torch)

    def rgb_to_gray(self) -> Image:
        """
        Return a grayscale Image obtained from an RGB Image

        :return: grayscale Image
        """
        return Image(self._rgb_to_gray())

    def _rgb_to_gray(self) -> torch.Tensor:
        """
        Return a grayscale 2D tensor obtained from an RGB 3D tensor

        :return: 2D tensor
        """
        return torchvision.transforms.v2.Grayscale()(self.torch)
        # return tf.image.rgb_to_grayscale(self.tensor)

    def warp_affine(self, affine_matrix: np.ndarray, in_place: bool = True) -> Image:
        """
        Transforms an image using an affine transform specified by a matrix

        :param affine_matrix: the affine transform matrix
        :param in_place: returns the same Image object if True
        :return:
        """
        tensor = torch.from_numpy(cv2.warpAffine(self.as_array(), affine_matrix, (self.shape[1], self.shape[0])))
        if not in_place:
            return Image(tensor)
        self.torch = tensor
        self.torch = self._convert_to_dtype(dtype=ImgDType.get_dtype(self._initial_torch.dtype))
        return self

    def resize(self, shape: tuple[int, int] | None = None, method: InterpolationMode = InterpolationMode.NEAREST,
               antialias: bool = True) -> Image:
        """
        Resize image to the defined shape with the defined method.

        :param shape: the target shape
        :param method: the resizing method

         * **bilinear: Bilinear interpolation.** If antialias is true, becomes a hat/tent filter function with radius 1 when downsampling.

         * **lanczos3: Lanczos kernel with radius 3.** High-quality practical filter but may have some ringing, especially on synthetic images.

         * **lanczos5: Lanczos kernel with radius 5.** Very-high-quality filter but may have stronger ringing.

         * **bicubic: Cubic interpolant of Keys.** Equivalent to Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel, particularly when upsampling.

         * **gaussian: Gaussian kernel** with radius 3, sigma = 1.5 / 3.0.

         * **nearest: (default) Nearest neighbour interpolation.** antialias has no effect when used with nearest neighbour interpolation.

         * **area: Anti-aliased resampling with area interpolation.** antialias has no effect when used with area interpolation; it always anti-aliases.

         * **mitchellcubic: Mitchell-Netravali Cubic non-interpolating filter.** For synthetic images (especially those lacking proper prefiltering), less ringing than Keys cubic kernel but less sharp.

        :return: the resized Image object
        """
        if shape is None or shape == self.shape[-2:]:
            return self
        return Image(v2.Resize(size=shape, interpolation=method)(self.torch))

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
            for c in range(self.torch.shape[0]):
                ax.hist(self.as_array()[c, ...].flatten(), bins=bins, histtype='step', color=colours[c])
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
        tensor = v2.functional.crop(self.torch, offset_height, offset_width, target_height, target_width)
        if new_image:
            return Image(tensor)
        self.torch = tensor
        return self

    def auto_contrast(self, preserve_tone: bool = True) -> Image:
        """
        Adjust contrast automatically using PIL package. RGBA images cannot be used here

        :param preserve_tone: if True, the tone is preserved
        :return: the current Image after correction
        """
        if self.torch.ndim == 3 and self.torch.size(0) == 4:
            rgb = torch.squeeze(v2.ToPureTensor()(v2.functional.autocontrast(tv.tv_tensors.Image(self.torch[:3]))))
            self.torch = torch.stack([rgb[0], rgb[1], rgb[2], self.torch[-1]], dim=-3)
        else:
            self.torch = torch.squeeze(v2.ToPureTensor()(v2.functional.autocontrast(tv.tv_tensors.Image(self.torch))))
        # self.tensor = tf.convert_to_tensor(
        #         np.array(ImageOps.autocontrast(PILimage.fromarray(self.as_array(ImgDType.uint8)), preserve_tone=preserve_tone)))
        return self

    def channel_last(self, dtype: ImgDType | None = None):
        """
        Converts the image tensor to channel-first format

        :param dtype: the desired dtype
        :return: the converted image tensor
        """
        tensor = self.torch if dtype is None else v2.ToDtype(dtype=dtype.torch_dtype, scale=True)(self.torch)
        if self.torch.ndim == 3:
            return torch.movedim(tensor, 0, 2)
        return tensor

    def adjust_contrast(self, factor: float = 2.0) -> Image:
        """
        Automatic contrast adjustment

        :param factor: the contrast adjustment factor
        :return: the current Image after correction
        """
        self.torch = v2.functional.adjust_contrast(self.torch, factor)
        return self

    def stretch_contrast(self, q: tuple[int, int] | None = None) -> Image:
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
            # self.tensor = tf.convert_to_tensor(exposure.rescale_intensity(img, in_range=(qlow, qhi)))
            self.torch = torch.tensor(exposure.rescale_intensity(img, in_range=(qlow, qhi)))
        return self

    def equalize_hist(self, adapt: bool = False) -> Image:
        """
        Adjust exposure using the histogram equalization method

        :param adapt: bool to set adaptative method
        :return: the current Image after correction
        """
        arr = self.as_array(ImgDType.float64)
        arr = arr / np.max(arr)
        if adapt:
            self.torch = torch.from_numpy(exposure.equalize_adapthist(arr))
        else:
            self.torch = torch.from_numpy(exposure.equalize_hist(arr))
        self.torch = v2.ToDtype(dtype=self._initial_torch.dtype)(self.torch)
        return self

    def sigmoid_correction(self) -> Image:
        """
        Exposure correction using the sigmoid method

        :return: the current Image after correction
        """
        self.torch = torch.from_numpy(exposure.adjust_sigmoid(self.as_array()))
        return self

    def decompose_channels(self) -> list[Image]:
        """
        Split an RGB image in a list of channels

        :return: list of one Image per channel
        """
        if len(self.shape) == 2:
            return [self]
        return [Image(array) for array in self.torch]

    @staticmethod
    def add(images: list[Image]) -> Image:
        """
        Pixelwise addition of images in a list

        :param images: the list of images
        :return: the resulting Image
        """
        return Image(torch.sum(torch.stack([i.torch for i in images]), dim=0))

    @staticmethod
    def mean(images: list[Image]) -> Image:
        """
        Compute the pixelwise mean of a list of images

        :param images: the list of images to average
        :return: the averaged image
        """
        if len(images) > 1:
            tensor = images[0].as_torch(ImgDType.float32)
            for i in images[1:]:
                tensor = torch.add(tensor, i.as_torch(ImgDType.float32))
            tensor = torch.div(tensor, len(images))
            return Image(v2.ToDtype(images[0].torch.dtype)(tensor))
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
            channels.append(Image(np.maximum(np.maximum(channels[0].as_array(), channels[1].as_array()), channels[2].as_array())))
        return Image(torch.stack([c.as_torch() for c in channels], dim=-3))
