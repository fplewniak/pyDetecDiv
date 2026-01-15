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
        if self.metrics is None:
            self.metrics = MetricCollection(metrics)
        else:
            self.metrics.add_metrics(metrics)


class ClassifierModelStats(ModelStats):
    def __init__(self, model_name = None, class_names = None, **kwargs):
        super().__init__(model_name = model_name, **kwargs)
        self.class_names = class_names if class_names is not None else []
        self.metrics = MetricCollection({'ConfusionMatrix_recall': MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='true'),
                                         'ConfusionMatrix_precision': MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='pred'),
                                         })

    @property
    def num_classes(self):
        return len(self.class_names)


class TrainingHistory:
    def __init__(self):
        self.loss = []
        self.val_loss = []
        self.metrics_values = []
        self.val_metrics_values = []
        self.main_metric = None
        self.best_epoch = 0

    @property
    def num_epochs(self):
        return len(self.loss)

    def metric_history(self, metric_name):
        return [d[metric_name].cpu() for d in self.metrics_values]

    def val_metric_history(self, metric_name):
        return [d[metric_name].cpu() for d in self.val_metrics_values]

    def plot(self, axs):
        axs.plot(self.loss)
        axs.plot(self.val_loss)
        axs.set_ylabel('Loss')
        axs.set_xlabel('epoch')
        axs.legend(['train', 'val'], loc='lower left')

    def plot_metric(self, axs, metric_name):
        axs.plot(self.metric_history(metric_name))
        axs.plot(self.val_metric_history(metric_name))
        axs.set_ylabel(metric_name)
        axs.set_xlabel('epoch')
        axs.legend(['train', 'val'], loc='upper left')


class TrainingStats(ModelStats):
    def __init__(self, model_name = None, **kwargs):
        super().__init__(model_name = model_name, **kwargs)
        self.history = TrainingHistory()
        self.val_metrics = None
        self.checkpoint_files = {'best': None, 'last': None}

    @property
    def main_metric(self):
        return self.history.main_metric

    @property
    def metrics_values(self):
        return self.history.metrics_values

    @property
    def val_metrics_values(self):
        return self.history.val_metrics_values

    @property
    def loss(self):
        return self.history.loss

    @property
    def val_loss(self):
        return self.history.val_loss

    def metric_history(self, metric_name=None):
        if metric_name is None:
            return {metric_name: self.history.metric_history(metric_name) for metric_name in self.metrics.keys()}
        return self.history.metric_history(metric_name)

    def val_metric_history(self, metric_name=None):
        if metric_name is None:
            return {metric_name: self.history.val_metric_history(metric_name) for metric_name in self.metrics.keys()}
        return self.history.val_metric_history(metric_name)

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

    def log_loss(self, loss):
        self.loss.append(loss)

    def log_val_loss(self, loss):
        self.val_loss.append(loss)


class ClassifierTrainingStats(TrainingStats, ClassifierModelStats):
    def __init__(self, model_name = None, class_names = None):
        super().__init__(model_name = model_name, class_names = class_names)
        print(self.class_names)
        self.val_metrics = MetricCollection({'ConfusionMatrix_recall': MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='true'),
                                         'ConfusionMatrix_precision': MulticlassConfusionMatrix(num_classes=len(self.class_names), normalize='pred'),
                                         })

    @property
    def num_classes(self):
        return len(self.class_names)
