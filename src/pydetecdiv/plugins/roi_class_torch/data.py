from datetime import datetime
import tables as tbl

import numpy as np


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
    return indices[:num_training], indices[num_training:num_validation + num_training], indices[num_validation + num_training:]
