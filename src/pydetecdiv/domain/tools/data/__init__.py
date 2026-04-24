"""
Classes and functions to handle data
"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import polars
import torch

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.domain.ROI import ROI


class RoiDataReader(ABC):
    """
    Abstract class defining the interface to access ROI data from any type of source (HDF5, NDTiff, etc.)
    """
    def __init__(self, source: Any):
        self.source = source

    @abstractmethod
    def roi_data(self, roi_idx: int | slice = None, frame: int | slice = 0) -> torch.Tensor:
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

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """
        Returns a list of class names
        :return: the list of class names
        """

    @abstractmethod
    def roi_id(self, roi_idx: int = None) -> int:
        """
        Returns the id_ of the desired ROI, designated by its index in the HDF5 file
        :param roi_idx: the index of the ROI in the HDF5 file
        :return: the ROI id_
        """

    @property
    @abstractmethod
    def roi_ids(self) -> list[int]:
        """
        Returns the list of all ROI IDs
        :return: all ROI ids in a list
        """

    @property
    @abstractmethod
    def num_rois(self) -> int:
        """
        Returns the number of ROIs in the source
        :return: the number of ROIs
        """

    @property
    @abstractmethod
    def num_frames(self) -> int:
        """
        Returns the number of frames in the source
        :return: the number of frames
        """

    @property
    @abstractmethod
    def num_targets(self) -> int:
        """
        Returns the number of targets in the source
        :return: the number of targets
        """

    def roi(self, roi_idx: int = None) -> ROI:
        """
        Returns the ROI object designated by its index in the HDF5 file
        :param roi_idx: the index of the ROI in the HDF5 file
        :return: the ROI object
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            return project.get_object('ROI', self.roi_id(roi_idx))

    def target_indices(self, roi_idx: int | list[int] = None) -> polars.DataFrame:
        """
        Returns all the indices of (ROI, frame) pairs in the source that have a target
        :return: the indices
        """
        if roi_idx is None:
            roi_idx = list(range(self.num_rois))
        elif isinstance(roi_idx, int):
            roi_idx = [roi_idx]
        rois, frames = [], []
        for idx in roi_idx:
            for frame in range(self.num_targets):
                if self.num_targets > 0 and self.target(idx, frame) > -1:
                    rois.append(idx)
                    frames.append(frame)
        return polars.DataFrame({'roi': rois, 'frame': frames})

    def all_indices(self, roi_idx: int | list[int] = None) -> polars.DataFrame:
        """
        Returns all the indices of (ROI, frame) pairs in the source
        :return: all the indices
        """
        if roi_idx is None:
            roi_idx = list(range(self.num_rois))
        elif isinstance(roi_idx, int):
            roi_idx = [roi_idx]
        rois, frames = [], []
        for idx in roi_idx:
            for frame in range(self.num_frames):
                rois.append(idx)
                frames.append(frame)
        return polars.DataFrame({'roi': rois, 'frame': frames})

    def no_target_indices(self, roi_idx: int | list[int] = None) -> polars.DataFrame:
        """
        Returns all the indices of (ROI, frame) pairs in the source that do not have any target
        :return: the indices
        """
        with_targets = self.target_indices(roi_idx)
        all_indices = self.all_indices(roi_idx)
        return all_indices.join(with_targets, left_on=['roi', 'frame'], right_on=['roi', 'frame'], how='anti')

    def close(self) -> None:
        """
        Close the data source if it has the close() method
        """
        if hasattr(self.source, 'close'):
            self.source.close()

def compute_class_weights(targets: np.ndarray) -> torch.Tensor:
    """
    Compute the weights for classes for class balancing

    :param targets: all the targets
    :return: the class weights
    """
    labels = targets.flatten()
    labels = labels[labels > -1]
    classes, class_counts = np.unique(labels, return_counts=True)
    total_counts = np.sum(class_counts)
    num_classes = len(class_counts)
    alpha = total_counts / class_counts
    return torch.tensor((num_classes * alpha) / np.sum(alpha), dtype=torch.float32)
