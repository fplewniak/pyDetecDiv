import math
import sys
from datetime import datetime
import tables as tbl

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from torchvision.transforms import transforms, v2
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


def prepare_data_for_training(hdf5_file: str, seqlen: int = 0, train: float = 0.6, validation: float = 0.2,
                              seed: int = 42) -> (list[int], list[int], list[int]):
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
    # total_counts = np.sum(class_counts)
    # class_weights = torch.tensor([total_counts / (len(classes) * c) for c in class_counts], dtype=torch.float32)
    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=labels), dtype=torch.float32)
    print(classes, file=sys.stderr)
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
    return indices[:num_training], indices[num_training:num_validation + num_training], indices[num_validation + num_training:], class_weights


class ROIDataset(Dataset):
    def __init__(self, h5file: str, indices: list, targets: bool = False, image_shape: tuple[int, int] = (60, 60), seqlen: int = 0,
                 seq2one: bool = True, transform: transforms = None):
        self.h5file = tbl.open_file(h5file, mode='r')
        self.roi_data = self.h5file.root.roi_data
        if targets:
            self.targets = self.h5file.root.targets
        else:
            self.targets = None
        self.class_names = self.h5file.root.class_names

        self.seqlen = seqlen
        self.seq2one = seq2one

        self.indices = indices
        self.image_shape = list(image_shape)

        # self.transform = transforms.Compose([v2.ToDtype(torch.float, scale=True),
        #                                     transforms.Normalize((0.5,), (0.5,)), ])
        self.transform = transforms.Compose([v2.ToDtype(torch.float, scale=True),
                                            v2.functional.autocontrast])
        # self.transform = v2.ToDtype(torch.float, scale=True)


        # self.transform = transform
        if transform:
            self.transform = transforms.Compose([self.transform, transform])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame, roi_id = self.indices[idx]
        if self.seqlen == 0:
            roi_data = torch.tensor(self.roi_data[frame, roi_id, ...]).permute(2, 0, 1)
            roi_data = F.resize(roi_data, size=self.image_shape)
            # targets = torch.zeros(len(self.class_names), dtype=torch.float32)
            # targets[self.targets[frame, roi_id]] = 1.0
            if self.targets:
                targets = torch.tensor(self.targets[frame, roi_id])
            if self.transform:
                roi_data = self.transform(roi_data)
        else:
            roi_data = torch.tensor(self.roi_data[frame:frame + self.seqlen, roi_id, ...]).permute(0, 3, 1, 2)
            if self.targets:
                if self.seq2one:
                    targets = torch.tensor(self.targets[frame+math.ceil(self.seqlen / 2.0), roi_id, ...])
                else:
                    targets = torch.tensor(self.targets[frame:frame+self.seqlen, roi_id, ...])
            # targets = torch.tensor(self.targets[frame + int(self.seqlen / 2.0), roi_id])
            # targets = torch.tensor(self.targets[frame, roi_id])
            # targets = torch.zeros(len(self.class_names), dtype=torch.float32)
            # targets[self.targets[frame, roi_id]] = 1.0
            # roi_data = F.resize(roi_data, size=self.image_shape)
            if self.transform:
                roi_data = torch.stack([self.transform(frame) for frame in roi_data], dim=0)
        if self.targets:
            return roi_data, targets
        return roi_data
