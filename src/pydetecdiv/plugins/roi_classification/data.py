"""
Some utility functions to handle data for ROI classification
"""
import math
import sys
from datetime import datetime
import tables as tbl

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import transforms, v2
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


def prepare_data_for_training(hdf5_file: str, seqlen: int = 0, train: float = 0.6, validation: float = 0.2,
                              seed: int = 42) -> tuple[list[list[int]], list[list[int]], list[list[int]], Tensor]:
    """
    Prepares data for training.

    :param hdf5_file: the HDF5 file containing the annotated ROI data
    :param seqlen: the length of ROI time sequence or 0 if the model takes a single image as input
    :param train: the proportion of images or sequences used for training
    :param validation: the proportion of images or sequences used for validation
    :param seed: the seed used to shuffle the data before dispatching data into the datasets
    :return: indices for training, validation, testing datasets
    """
    h5file = tbl.open_file(hdf5_file, mode='r')
    print(f'{datetime.now().strftime("%H:%M:%S")}: Reading targets into a numpy array')
    targets_arr = h5file.root.targets
    num_frames = targets_arr.shape[0]
    num_rois = targets_arr.shape[1]
    targets = targets_arr.read()
    labels = targets.flatten()
    labels = labels[labels > -1]
    classes, class_counts = np.unique(labels, return_counts=True)
    print(classes, file=sys.stderr)
    total_counts = np.sum(class_counts)
    num_classes = len(class_counts)
    alpha = total_counts / class_counts
    class_weights = torch.tensor((num_classes * alpha) / np.sum(alpha), dtype=torch.float32)
    print(class_weights, file=sys.stderr)
    print(f'{datetime.now().strftime("%H:%M:%S")}: Select valid targets from array with shape {targets.shape}')
    if seqlen == 0:
        indices = [[frame, roi] for roi in range(num_rois) for frame in range(num_frames) if targets[frame, roi] != -1]
    else:
        indices = [[frame, roi] for roi in range(num_rois) for frame in range(0, num_frames - seqlen + 1, seqlen)
                   if np.all([targets[frame:frame + seqlen, roi] != -1])]
    print(f'{datetime.now().strftime("%H:%M:%S")}: Kept {len(indices)} valid ROI frames or sequences')

    print(f'{datetime.now().strftime("%H:%M:%S")}: Shuffling data')
    rng = np.random.default_rng(seed)
    rng.shuffle(np.array(indices))
    print(f'{datetime.now().strftime("%H:%M:%S")}: Determine training and validation datasets size')
    num_training = int(len(indices) * train)
    num_validation = int(len(indices) * validation)
    print(f'{datetime.now().strftime("%H:%M:%S")}: Close HDF5 file and return datasets indices')
    h5file.close()
    return (indices[:num_training], indices[num_training:num_validation + num_training],
            indices[num_validation + num_training:], class_weights)


def prepare_data_for_inference(hdf5_file: str, seqlen: int = 0) -> list[tuple[int, int]]:
    """
    Prepare data for class prediction

    :param hdf5_file: the HDF5 file containing the unannotated ROI data
    :param seqlen: the sequence length
    :return: a list of indices in the HDF5 arrays for the dataset
    """
    h5file = tbl.open_file(hdf5_file, mode='r')
    print(f'{datetime.now().strftime("%H:%M:%S")}: Getting valid indices for unannotated ROIs frames')
    num_rois = h5file.root.roi_ids.shape[0]
    num_frames = h5file.root.num_frames

    if seqlen == 0:
        indices = [(frame, roi) for roi in range(num_rois) for frame in range(num_frames[roi])]
    else:
        indices = [(frame, roi) for roi in range(num_rois) for frame in range(0, num_frames[roi] - seqlen + 1, seqlen)]

    print(f'{datetime.now().strftime("%H:%M:%S")}: Kept {len(indices)} valid ROI frames or sequences')
    h5file.close()
    return indices


class ROIDataset(Dataset):
    """
    Dataset returning (ROI data, frame) pairs
    """
    def __init__(self, h5file: str, indices: list, targets: bool = False, image_shape: tuple[int, int] = (60, 60), seqlen: int = 0,
                 seq2one: bool = True, transform: torch.nn.Module = None):
        self.h5file = tbl.open_file(h5file, mode='r')
        self.roi_data = self.h5file.root.roi_data
        self.roi_ids = self.h5file.root.roi_ids

        if targets:
            self.targets = self.h5file.root.targets
            self.class_names = self.h5file.root.class_names
        else:
            self.targets = None
            self.class_names = None

        self.seqlen = seqlen
        self.seq2one = seq2one

        self.indices = indices
        self.image_shape = list(image_shape)

        # self.transform = transforms.Compose([v2.ToDtype(torch.float, scale=True),
        #                                     transforms.Normalize((0.5,), (0.5,)), ])
        # self.transform = transforms.Compose([v2.ToDtype(torch.float, scale=True),
        #                                     v2.functional.autocontrast])
        self.transform = v2.ToDtype(torch.float, scale=True)


        # self.transform = transform
        if transform:
            self.transform = transforms.Compose([self.transform, transform])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        frame, mapping = self.indices[idx]
        roi_id = self.roi_ids[mapping]
        if self.seqlen == 0:
            roi_data = torch.tensor(self.roi_data[frame, mapping, ...]).permute(2, 0, 1)
            roi_data = F.resize(roi_data, size=self.image_shape)
            # targets = torch.zeros(len(self.class_names), dtype=torch.float32)
            # targets[self.targets[frame, roi_id]] = 1.0
            if self.targets:
                targets = torch.tensor(self.targets[frame, mapping])
            if self.transform:
                roi_data = self.transform(roi_data)
        else:
            roi_data = torch.tensor(self.roi_data[frame:frame + self.seqlen, mapping, ...]).permute(0, 3, 1, 2)
            if self.targets:
                if self.seq2one:
                    targets = torch.tensor(self.targets[frame+math.ceil(self.seqlen / 2.0), mapping, ...])
                else:
                    targets = torch.tensor(self.targets[frame:frame+self.seqlen, mapping, ...])
            # targets = torch.tensor(self.targets[frame + int(self.seqlen / 2.0), roi_id])
            # targets = torch.tensor(self.targets[frame, roi_id])
            # targets = torch.zeros(len(self.class_names), dtype=torch.float32)
            # targets[self.targets[frame, roi_id]] = 1.0
            # roi_data = F.resize(roi_data, size=self.image_shape)
            if self.transform:
                roi_data = torch.stack([self.transform(frame) for frame in roi_data], dim=0)
        if self.targets:
            return roi_data, targets
        return roi_data, frame, roi_id

    # def get_roi_name(self, idx):
    #     _, roi_id = self.indices[idx]
    #     return self.roi_names[roi_id][0].decode()

    def get_roi_id(self, idx: int) -> int:
        """
        Get the ROI id from the data index

        :param idx: the data index
        :return: the ROI id
        """
        _, mapping = self.indices[idx]
        return self.roi_ids[mapping]

    def get_frame(self, idx: int) -> int:
        """
        Get the frame for the data index

        :param idx: the data index
        :return: the corresponding frame
        """
        frame, _ = self.indices[idx]
        return frame

    def close(self) -> None:
        """
        Close the dataset
        """
        self.h5file.close()
