import numpy as np
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix

class ModelStats:
    def __init__(self, model_name = None, **kwargs):
        super().__init__(**kwargs)
        self.metrics = None
        self.model_name = model_name
        self.checkpoint = None

    def load_model(self, checkpoint = None, device = 'cpu'):
        if checkpoint is not None:
            self.checkpoint = checkpoint
        if self.checkpoint is not None:
            return torch.jit.load(self.checkpoint).to(device)
        return None

    def add_metrics(self, metrics):
        self.metrics.add_metrics(metrics)


class ClassifierModelStats(ModelStats):
    def __init__(self, model_name = None, class_names = None, **kwargs):
        super().__init__(model_name = model_name, **kwargs)
        self.class_names = class_names if class_names is not None else []
        self.metrics = MetricCollection({'recall': MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='true'),
                                         'precision': MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='pred'),
                                         })

    @property
    def num_classes(self):
        return len(self.class_names)


class TrainingHistory:
    def __init__(self):
        self.train = {'loss': []}
        self.val = {'loss': []}
        self.best_epoch = 0

    def extend(self, metrics):
        for metric_name, metric_value in metrics['train'].items():
            if metric_name not in self.train:
                self.train[metric_name] = [metric_value]
            else:
                self.train[metric_name].append(metric_value)
        for metric_name, metric_value in metrics['val'].items():
            if metric_name not in self.val:
                self.val[metric_name] = [metric_value]
            else:
                self.val[metric_name].append(metric_value)


    def plot(self, axs, metric_name):
        axs.plot(self.train[metric_name])
        axs.plot(self.val[metric_name])
        axs.set_ylabel(metric_name)
        axs.set_xlabel('epoch')
        axs.legend(['train', 'val'], loc='lower right')


class TrainingStats(ModelStats):
    def __init__(self, model_name = None, **kwargs):
        super().__init__(model_name = model_name, **kwargs)
        self.history = TrainingHistory()
        self.val_metrics = None
        self.metrics_values = []
        self.val_metrics_values = []
        self.evaluation = {}
        self.checkpoint_files = {'best': None, 'last': None}

    def load_model(self, checkpoint = 'best', device = 'cpu'):
        return super().load_model(checkpoint = self.checkpoint_files[checkpoint], device = device)
        # if self.checkpoint_files[checkpoint] is not None:
        #     return torch.jit.load(self.checkpoint_files[checkpoint]).to(device)
        # return None

    def add_metrics(self, metrics):
        self.metrics.add_metrics(metrics)
        self.val_metrics.add_metrics(metrics)

    def log_metrics(self):
        self.metrics_values.append(self.metrics.compute())

    def log_val_metrics(self):
        self.val_metrics_values.append(self.val_metrics.compute())


class ClassifierTrainingStats(TrainingStats, ClassifierModelStats):
    def __init__(self, model_name = None, class_names = None):
        super().__init__(model_name = model_name, class_names = class_names)
        print(self.class_names)
        self.val_metrics = MetricCollection({'recall': MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='true'),
                                         'precision': MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='pred'),
                                         })

    @property
    def num_classes(self):
        return len(self.class_names)
