"""
Classes defining metrics to assess training
"""
import sys
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from torchmetrics import Metric


class NWScore(Metric):
    """
    Needleman & Wunsch score. This metric first aligns the prediction to the ground truth sequence, allowing gaps and returns the
    corresponding alignment score relative to the maximum scores that can be achieved for both sequences
    """
    MATCH = 1
    DEL1 = 2
    DEL2 = 3

    m = 1.0
    mm = -2.0
    gop = -1.0
    gep = -1.5

    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, num_classes=6, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.score = 0.0
        self.updated = False

        self.simmatrix = np.zeros([num_classes, num_classes], dtype=np.float64)
        for i in range(num_classes):
            for j in range(num_classes):
                self.simmatrix[i, j] = self.m if i == j else self.mm

    def update(self, preds: Tensor, target: Tensor) -> None:
        for seq in preds:
            self.preds.append(seq.argmax(dim=0))
        for seq in target:
            self.target.append(seq)
        self.updated = True
        # self.preds.append(torch.from_numpy(np.array([p.argmax(dim=0) for p in preds.cpu()])))
        # self.target.append(target)

    def nw_align(self, seq1, seq2):
        compath = np.zeros([len(seq1) + 1, len(seq2) + 1])
        edition_matrix = np.zeros([len(seq1) + 1, len(seq2) + 1])
        for i in range(1, len(seq1) + 1):
            compath[i, 0] = self.gop + (i - 1) * self.gep
            edition_matrix[i, 0] = self.DEL2

        for i in range(1, len(seq2) + 1):
            compath[0, i] = self.gop + (i - 1) * self.gep
            edition_matrix[0, i] = self.DEL1

        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                gp1 = self.gep if edition_matrix[i, j - 1] == self.DEL1 else self.gop
                gp2 = self.gep if edition_matrix[i - 1, j] == self.DEL2 else self.gop

                aln = compath[i - 1, j - 1] + self.simmatrix[seq1[i - 1], seq2[j - 1]]
                del1 = compath[i, j - 1] + gp1
                del2 = compath[i - 1, j] + gp2

                match np.argmax([aln, del1, del2]):
                    case 0:
                        edition_matrix[i, j] = self.MATCH
                        compath[i, j] = aln
                    case 1:
                        edition_matrix[i, j] = self.DEL1
                        compath[i, j] = del1
                    case 2:
                        edition_matrix[i, j] = self.DEL2
                        compath[i, j] = del2
        return compath[-1, -1]

    def compute(self):
        if self.updated:
            self.score = []
            for preds, target in zip(self.preds, self.target):
                seq_p, seq_t = preds.detach().cpu(), target.detach().cpu()
                # self.score.append(2 * self.nw_align(seq_p, seq_t) / (self.nw_align(seq_p, seq_p) + self.nw_align(seq_t, seq_t)))
                self.score.append(2 * self.nw_align(seq_p, seq_t) / (self.MATCH * (len(seq_p) + len(seq_t))))
            self.score = np.mean(self.score)
        return torch.tensor(self.score)

#
# class AccuracyByClass(Metric):
#     """
#     Accuracy metric computing accuracy for each class and then reducing it if requested, in order to mitigate the effect of
#     unbalanced classes
#     """
#     name = 'class accuracy'
#     def __init__(self, reduction: str = 'mean'):
#         super().__init__()
#         self.reduction = reduction
#         self.reset_sampling()
#
#     def get_value(self) -> Tensor | None:
#         """
#         Return the accuracy value computed by class
#         """
#         outputs = self.sample_outputs
#         targets = self.sample_targets
#         if outputs.dim() == 1:
#             return None
#         if outputs.dim() == 2:
#             N, C = outputs.size(0), outputs.size(1)
#             outputs = outputs.view(N, C, -1)
#             targets = targets.view(N, -1)
#         else:
#             N, T, C = outputs.size(0), outputs.size(1), outputs.size(2)
#             outputs = outputs.view(N * T, C, -1)
#             targets = targets.view(N * T, -1)
#         pred = torch.argmax(outputs, dim=1)
#         correct = (pred == targets).unsqueeze(1)
#         target_onehot = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=C).transpose(1, 2)
#         correct = correct.expand_as(target_onehot) * target_onehot
#         total = target_onehot.sum((0, 2))
#         correct = correct.sum((0, 2)) / total
#         match self.reduction:
#             case 'mean':
#                 return correct.mean()
#             case 'sum':
#                 return correct.sum()
#             case None:
#                 return correct
