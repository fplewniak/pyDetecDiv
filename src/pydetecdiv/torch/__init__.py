import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix


class TrainingHistory:
    def __init__(self):
        self.train = {}
        self.val = {}
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


class TrainingStats:
    def __init__(self, model_name = None):
        self.history = TrainingHistory()
        self.metrics = None
        self.metrics_values = []
        self.evaluation = {}
        self.model_name = model_name
        self.checkpoint_files = {'best': None, 'last': None}

    def load_model(self, checkpoint = 'best', device = 'cpu'):
        if self.checkpoint_files[checkpoint] is not None:
            return torch.jit.load(self.checkpoint_files[checkpoint]).to(device)
        return None

    def add_metrics(self, metrics):
        self.metrics.add_metrics(metrics)

    def log_metrics(self):
        self.metrics_values.append(self.metrics.compute())


class ClassifierTrainingStats(TrainingStats):
    def __init__(self, model_name = None, class_names = None):
        super().__init__(model_name = model_name)
        self.class_names = class_names if class_names is not None else []
        self.metrics = MetricCollection({'recall': MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='true'),
                                         'precision': MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='pred'),
                                         })

    @property
    def num_classes(self):
        return len(self.class_names)
