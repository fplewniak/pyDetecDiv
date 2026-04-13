"""
Classes defining metrics to assess training
"""
import sys
from typing import Optional

import numpy as np
import torch
from pyseq_align import NeedlemanWunsch
from torch import Tensor

from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (MulticlassMatthewsCorrCoef, MulticlassF1Score, MulticlassAUROC, MulticlassAccuracy,
                                         MulticlassAveragePrecision, MulticlassCalibrationError, MulticlassPrecision,
                                         MulticlassRecall)


class NWScore(Metric):
    MATCH = 2
    MISSMATCH = -4
    GOP = -2
    GEP = -2
    NEUTRAL = 0

    SUB_MATRIX = {
        '0': {'0': MATCH, '1': MISSMATCH, '2': MISSMATCH, '3': MISSMATCH, '4': MISSMATCH, '5': MISSMATCH},
        '1': {'0': MISSMATCH, '1': MATCH, '2': MISSMATCH, '3': MISSMATCH, '4': MISSMATCH, '5': MISSMATCH},
        '2': {'0': MISSMATCH, '1': MISSMATCH, '2': MATCH, '3': MISSMATCH, '4': MISSMATCH, '5': MISSMATCH},
        '3': {'0': MISSMATCH, '1': MISSMATCH, '2': MISSMATCH, '3': MATCH, '4': MISSMATCH / 2, '5': MISSMATCH / 2},
        '4': {'0': MISSMATCH, '1': MISSMATCH, '2': MISSMATCH, '3': NEUTRAL, '4': MATCH, '5': MISSMATCH},
        '5': {'0': MISSMATCH, '1': MISSMATCH, '2': MISSMATCH, '3': NEUTRAL, '4': NEUTRAL, '5': MATCH},
        }

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

    def update(self, preds: Tensor, target: Tensor) -> None:
        for seq in preds:
            self.preds.append(seq.argmax(dim=0).cpu())
        for seq in target:
            self.target.append(seq.cpu())
        self.updated = True

    def compute(self):
        if self.updated:
            self.score = np.mean([self.nw_align(target, preds) for preds, target in zip(self.preds, self.target)])
        return torch.tensor(self.score)

    def nw_align(self, seq1, seq2):
        str1 = ''.join([str(c.item()) for c in seq1])
        str2 = ''.join([str(c.item()) for c in seq2])
        nw = NeedlemanWunsch(match=self.MATCH, mismatch=self.MISSMATCH, gap_open=self.GOP, gap_extend=self.GEP,
                             substitution_matrix=self.SUB_MATRIX)
        al = nw.align(str1, str2)
        # print(al.result_a, file=sys.stderr)
        # print(al.result_b, al.score * 2 / ((len(str1) + len(str2)) * self.MATCH), file=sys.stderr)
        return al.score * 2 / ((len(str1) + len(str2)) * self.MATCH)

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

def is_single_value_metric(metric_name: str) -> bool:
    return metric_name not in ['ROC', 'PRC', 'ConfusionMatrix_recall', 'ConfusionMatrix_precision']

def set_metrics(num_classes: int) -> MetricCollection:
    """
    Set the metrics collection for the run

    :param num_classes: the number of classes
    :return: the metrics collection
    """
    metrics = MetricCollection([
        MetricCollection({'MCC': MulticlassMatthewsCorrCoef(num_classes=num_classes)}),
        MetricCollection({'F1score': MulticlassF1Score(num_classes=num_classes, average='weighted')}),
        MetricCollection({'AUROC': MulticlassAUROC(num_classes=num_classes)}),
        MetricCollection({'Accuracy': MulticlassAccuracy(num_classes=num_classes)}),
        MetricCollection({'AUPRC': MulticlassAveragePrecision(num_classes=num_classes)}),
        MetricCollection({'Calibration Error': MulticlassCalibrationError(num_classes=num_classes)}),
        # MetricCollection({'Negative Predictive Value': MulticlassNegativePredictiveValue(num_classes=num_classes)}),
        MetricCollection({'Precision': MulticlassPrecision(num_classes=num_classes)}),
        MetricCollection({'Recall': MulticlassRecall(num_classes=num_classes)}),
        # MetricCollection({'NWScore': NWScore(num_classes=num_classes)}),
        # MetricCollection({'Specificity': MulticlassSpecificity(num_classes=num_classes)}),
        ])
    return metrics
