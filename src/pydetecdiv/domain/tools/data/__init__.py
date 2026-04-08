from abc import ABC, abstractmethod
from typing import Any

import torch


class RoiDataReader(ABC):
    """
    Abstract class defining the interface to access ROI data from any type of source (HDF5, NDTiff, etc.)
    """
    def __init__(self, source: Any):
        self.source = source

    @abstractmethod
    def roi_data(self, roi_idx: int = None, frame: int = 0) -> torch.Tensor:
        """
        Returns data for the ROI specified by `roi_idx` and `frame`. Should return a sequence only if targets apply when using a
        many-to-one classifier. For many-to-may classifiers, the method should return a single ROI data and it should be the
        responsibility of dataset class expanding torch.utils.data.Dataset to build the sequence

        :param roi_idx: the index for the desired ROI
        :param frame: the desired frame (first frame for a sequence)
        :return: a tensor of ROI data, (C, H, W) for a single ROI or (T, C, H, W) for a sequence of ROIs.
        """

    @abstractmethod
    def target(self, roi_idx: int = None, frame: int = 0) -> torch.Tensor:
        """
        Returns target data for the ROI specified by `roi_idx` and `frame`.
        :param roi_idx: the index for the desired ROI
        :param frame: the desired frame (first frame for a sequence)
        :return: a 0D Tensor of target data
        """

    @abstractmethod
    def class_names(self) -> list[str]:
        """
        Returns a list of class names
        :return: the list of class names
        """
