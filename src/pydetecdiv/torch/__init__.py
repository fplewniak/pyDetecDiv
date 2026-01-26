"""
A module with torch specific helper classes
"""
import sys
from typing import Literal

import matplotlib.axes
from torch import jit, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassROC, MulticlassPrecisionRecallCurve


def is_single_value_metric(metric_name: str) -> bool:
    return metric_name not in ['ROC', 'PRC', 'ConfusionMatrix_recall', 'ConfusionMatrix_precision']


class ModelStats:
    """
    A generic class to handle model statistics
    """

    def __init__(self, model_name=None, **kwargs):
        super().__init__(**kwargs)
        self.metrics = None
        self.model_name = model_name
        self.checkpoint = None

    def load_model(self, checkpoint: Literal['best', 'last'] = None,
                   device: Literal['cpu', 'gpu'] = 'cpu') -> nn.Module | None:
        """
        Loads a PyTorch model from a checkpoint.

        :param checkpoint: the checkpoint to load
        :param device: the device to load the model on
        :return: the model or None
        """
        if checkpoint is not None:
            self.checkpoint = checkpoint
        if self.checkpoint is not None:
            return jit.load(self.checkpoint).to(device)
        return None

    def add_metrics(self, metrics: MetricCollection):
        """
        Add metrics to compute

        :param metrics: the metrics to add
        """
        if self.metrics is None:
            self.metrics = MetricCollection(metrics)
        else:
            self.metrics.add_metrics(metrics)


class ClassifierModelStats(ModelStats):
    """
    A generic class to handle statistics for a classifier model.
    """

    def __init__(self, model_name: str = None, class_names: list[str] = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.class_names = class_names if class_names is not None else []
        self.metrics = MetricCollection({'ConfusionMatrix_recall'   : MulticlassConfusionMatrix(num_classes=len(self.class_names),
                                                                                                normalize='true'),
                                         'ConfusionMatrix_precision': MulticlassConfusionMatrix(num_classes=len(self.class_names),
                                                                                                normalize='pred'),
                                         'ROC'                      : MulticlassROC(num_classes=len(self.class_names), ),
                                         'PRC'                      : MulticlassPrecisionRecallCurve(
                                             num_classes=len(self.class_names), )
                                         })

    @property
    def num_classes(self) -> int:
        """
        The number of classes for the classifier

        :return: the number of classes
        """
        return len(self.class_names)


class TrainingHistory:
    """
    A generic class for handling training history
    """

    def __init__(self):
        self.loss = []
        self.val_loss = []
        self.metrics_values = []
        self.val_metrics_values = []
        self.main_metric: str|None = None
        self.best_epoch = 0

    @property
    def num_epochs(self) -> int:
        """
        The number of epoch in history

        :return: the number of epochs
        """
        return len(self.loss)

    def metric_history(self, metric_name):
        """
        The history of the requested metric for the training dataset

        :param metric_name: the metric name
        :return: the metric history
        """
        return [d[metric_name].cpu() for d in self.metrics_values]

    def val_metric_history(self, metric_name: str) -> list[float]:
        """
        The history of the requested metric for the validation dataset

        :param metric_name: the metric name
        :return: the metric history
        """
        return [d[metric_name].cpu() for d in self.val_metrics_values]

    def plot(self, axs: matplotlib.axes.Axes) -> None:
        """
        Plot the loss history for training and validation and training datasets in a matplotlib figure.

        :param axs: the matplotlib axis
        """
        axs.plot(self.loss)
        axs.plot(self.val_loss)
        axs.set_ylabel('Loss')
        axs.set_xlabel('epoch')
        axs.legend(['train', 'val'], loc='lower left')

    def plot_metric(self, axs: matplotlib.axes.Axes, metric_name: str) -> None:
        """
        Plot a metric history for training and validation and training datasets in a matplotlib figure.

        :param axs: the matplotlib axis
        :param metric_name: the name of the metric
        """
        axs.plot(self.metric_history(metric_name))
        axs.plot(self.val_metric_history(metric_name))
        axs.set_ylabel(metric_name)
        axs.set_xlabel('epoch')
        axs.legend(['train', 'val'], loc='upper left')


class TrainingStats(ModelStats):
    """
    A general class for logging training stats for any model
    """

    def __init__(self, model_name: str | None = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.history = TrainingHistory()
        self.val_metrics = None
        self.checkpoint_files = {'best': None, 'last': None}

    @property
    def main_metric(self) -> str:
        """
        The followed metric

        :return: the metric object
        """
        return self.history.main_metric

    @property
    def metrics_values(self):
        """
        The history of the followed metric for training dataset

        :return: the history of the followed metric
        :return:
        """
        return self.history.metrics_values

    @property
    def val_metrics_values(self) -> list[float]:
        """
        The history of the followed metric for validation dataset

        :return: the history of the followed metric
        """
        return self.history.val_metrics_values

    @property
    def loss(self) -> list[float]:
        """
        Return the loss history for the training dataset

        :return: the loss history
        """
        return self.history.loss

    @property
    def val_loss(self) -> list[float]:
        """
        Return the loss history for the validation dataset

        :return: the loss history
        """
        return self.history.val_loss

    def metric_history(self, metric_name: str = None) -> list[float] | dict[str, list[float]]:
        """
        Return the metric history for the training dataset

        :param metric_name: the metric name
        :return: the metric history
        """
        if metric_name is None:
            return {metric_name: self.history.metric_history(metric_name) for metric_name in self.metrics.keys()}
        return self.history.metric_history(metric_name)

    def val_metric_history(self, metric_name: str = None) -> list[float] | dict[str, list[float]]:
        """
        Return the metric history for the validation dataset

        :param metric_name: the metric name
        :return: the metric history
        """
        if metric_name is None:
            return {metric_name: self.history.val_metric_history(metric_name) for metric_name in self.metrics.keys()}
        return self.history.val_metric_history(metric_name)

    def best_value(self, metric_name: str, validation: bool = True):
        """
        Get the best value for a metric and the corresponding epoch

        :param metric_name: the name of the metric
        :param validation: whether to return the validation metric or the training metric
        :return:
        """
        values = self.history.val_metric_history(metric_name) if validation else self.history.metric_history(metric_name)

        if self.metrics[metric_name].higher_is_better:
            best_val = max(values)
        else:
            best_val = min(values)
        epoch = values.index(best_val)
        return epoch, best_val.item()

    def load_model(self, checkpoint: Literal['best', 'last'] = 'best', device: Literal['cpu', 'gpu'] = 'cpu'):
        """
        Loads the model for the requested checkpoint
        :param checkpoint:
        :param device:
        :return:
        """
        return super().load_model(checkpoint=self.checkpoint_files[checkpoint], device=device)
        # if self.checkpoint_files[checkpoint] is not None:
        #     return torch.jit.load(self.checkpoint_files[checkpoint]).to(device)
        # return None

    def add_metrics(self, metrics: MetricCollection) -> None:
        """
        Adds a collection of metrics to the metrics that will be collected during training for both training and validation datasets

        :param metrics: the metrics collection
        """
        self.metrics.add_metrics(metrics)
        self.val_metrics.add_metrics(metrics)

    def log_metrics(self) -> None:
        """
        Appends the metrics values to the training metrics history
        """
        self.metrics_values.append(self.metrics.compute())

    def log_val_metrics(self) -> None:
        """
        Appends the metrics values to the validation metrics history
        """
        self.val_metrics_values.append(self.val_metrics.compute())

    def log_loss(self, loss: float) -> None:
        """
        Appends the loss value to the training loss history

        :param loss: the loss value
        """
        self.loss.append(loss)

    def log_val_loss(self, loss: float) -> None:
        """
        Appends the loss value to the validation loss history

        :param loss: the loss value
        """
        self.val_loss.append(loss)


class ClassifierTrainingStats(TrainingStats, ClassifierModelStats):
    """
    A class to handle training statistics for a classifier model
    """

    def __init__(self, model_name=None, class_names=None):
        super().__init__(model_name=model_name, class_names=class_names)
        self.val_metrics = MetricCollection({'ConfusionMatrix_recall'   :
                                                 MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='true'),
                                             'ConfusionMatrix_precision':
                                                 MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='pred'),
                                             })

    @property
    def num_classes(self) -> int:
        """
        The number of classes
        """
        return len(self.class_names)
