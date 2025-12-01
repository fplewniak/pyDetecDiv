"""
Classes defining metrics to assess training
"""
import torch
from torch import Tensor


class Metrics:
    """
    Abstract class defining metrics
    """
    def __init__(self):
        self.sample_outputs = None
        self.sample_targets = None

    @property
    def value(self) -> Tensor | None:
        """
        The value of the metrics, moved to the cpu
        """
        metrics_value = self.get_value()
        return metrics_value.cpu()

    def sampling(self, outputs: Tensor, targets: Tensor):
        """
        Samples outputs and corresponding ground truth for computing the metrics
        :param outputs: the predicted outputs
        :param targets: the ground truth
        """
        if self.sample_outputs.device != outputs.device:
            self.sample_outputs = self.sample_outputs.to(outputs.device)
        self.sample_outputs = torch.cat([self.sample_outputs, outputs])
        if self.sample_targets.device != targets.device:
            self.sample_targets = self.sample_targets.to(outputs.device)
        self.sample_targets = torch.cat([self.sample_targets, targets])

    def reset_sampling(self):
        """
        Reset the samples to be used for metrics computation
        """
        self.sample_outputs = torch.tensor([])
        self.sample_targets = torch.tensor([])

    def get_value(self) -> Tensor | None:
        """
        Abstract method to return the metrics value
        """
        pass

class Accuracy(Metrics):
    """
    Accuracy metrics
    """
    def __init__(self, weighted_mean: bool = True):
        super().__init__()
        self.weighted_mean = weighted_mean
        self.reset_sampling()

    def get_value(self) -> Tensor | None:
        """
        Return the accuracy value
        """
        outputs = self.sample_outputs
        targets = self.sample_targets
        if outputs.dim() == 1:
            return None
        if outputs.dim() == 2:
            N, C = outputs.size(0), outputs.size(1)
            outputs = outputs.view(N, C, -1)
            targets = targets.view(N, -1)
        else:
            N, T, C = outputs.size(0), outputs.size(1), outputs.size(2)
            outputs = outputs.view(N * T, C, -1)
            targets = targets.view(N * T, -1)
        pred = torch.argmax(outputs, dim=1)
        correct = pred == targets
        return correct.sum() / len(correct)


class AccuracyByClass(Metrics):
    """
    Accuracy metrics computing accuracy for each class and then reducing it if requested, in order to mitigate the effect of
    unbalanced classes
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.reset_sampling()

    def get_value(self) -> Tensor | None:
        """
        Return the accuracy value computed by class
        """
        outputs = self.sample_outputs
        targets = self.sample_targets
        if outputs.dim() == 1:
            return None
        if outputs.dim() == 2:
            N, C = outputs.size(0), outputs.size(1)
            outputs = outputs.view(N, C, -1)
            targets = targets.view(N, -1)
        else:
            N, T, C = outputs.size(0), outputs.size(1), outputs.size(2)
            outputs = outputs.view(N * T, C, -1)
            targets = targets.view(N * T, -1)
        pred = torch.argmax(outputs, dim=1)
        correct = (pred == targets).unsqueeze(1)
        target_onehot = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes=C).transpose(1, 2)
        correct = correct.expand_as(target_onehot) * target_onehot
        total = target_onehot.sum((0, 2))
        correct = correct.sum((0, 2)) / total
        match self.reduction:
            case 'mean':
                return correct.mean()
            case 'sum':
                return correct.sum()
            case None:
                return correct
