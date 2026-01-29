"""
GUI for training and fine tuning models
"""
import gc
import random
import sys

import matplotlib.axes
import numpy as np
import torch
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QFrame, QGridLayout
import pyqtgraph as pg

import pydetecdiv.plugins
from pydetecdiv.app import StdoutWaitDialog, PyDetecDiv
from pydetecdiv.app.gui.core import Colours
from pydetecdiv.app.gui.core.widgets.viewers.plots import MatplotViewer, ChartView

from pydetecdiv.plugins.gui import (ComboBox, AdvancedButton, SpinBox, ParametersFormGroupBox, DoubleSpinBox,
                                    RadioButton, set_connections, Label, Dialog)
from pydetecdiv.plugins.roi_classification.data import ROIDataset
from pydetecdiv.torch import ClassifierTrainingStats, TrainingHistory, is_single_value_metric


class TuneHyperparamDialog(Dialog):
    """
    A Dialog window to specify hyperparameters for training a model
    """
    job_finished: Signal = Signal(object)

    def __init__(self, plugin: pydetecdiv.plugins.Plugin, title: str = None):
        super().__init__(plugin, title='Tune hyperparameters for classification model')
        self.button_box = self.addButtonBox()
        self.arrangeWidgets([self.button_box])

        set_connections({self.button_box.accepted: self.wait_for_tunning,
                         self.button_box.rejected: self.close
                         })

        self.fit_to_contents()
        self.exec()

    def wait_for_tunning(self) -> None:
        """
        Open a waiting dialog window to wait for completion of hyperparameters tuning job
        """
        wait_dialog = StdoutWaitDialog('**Tuning hyperparameters**', self)
        wait_dialog.resize(500, 300)
        self.job_finished.connect(wait_dialog.stop_redirection)
        self.job_finished.connect(self.show_hyperparameters)
        wait_dialog.wait_for(self.run_tuning)
        self.close()

    def run_tuning(self) -> None:
        """
        Run a model training job
        """
        print('Run_tuning method', file=sys.stderr)
        self.job_finished.emit(self.plugin.tune_hyperparameters())

    def show_hyperparameters(self, trial):
        """
        Displays the hyperparameters for the specified trial
        :param trial:
        """
        print("  Params: ")
        for key, value in trial.params.items():
            print(f'    {key}: {value}')


class TrainingDialog(Dialog):
    """
    A Dialog window to specify hyperparameters for training a model
    """
    job_finished: Signal = Signal(object)

    def __init__(self, plugin: pydetecdiv.plugins.Plugin, title: str = None):
        super().__init__(plugin, title='Training classification model')

        self.classifier_selection = self.addGroupBox('Classifier')
        self.classifier_selection.addOption('Network', ComboBox, parameter=self.plugin.parameters['model'])
        self.classifier_selection.addOption('Classes', ComboBox, parameter=self.plugin.parameters['class_names'])
        self.classifier_selection.addOption('LSTM hidden size', SpinBox, parameter=self.plugin.parameters['hidden_size'])
        self.classifier_selection.addOption('LSTM layer number', SpinBox, parameter=self.plugin.parameters['num_layers'])
        self.classifier_selection.addOption('Dropout', DoubleSpinBox, parameter=self.plugin.parameters['dropout'])

        self.hyper = self.addGroupBox('Hyper parameters')
        self.hyper.addOption('Epochs:', SpinBox, adaptive=True, parameter=self.plugin.parameters['epochs'])

        self.hyper.addOption('Batch size:', SpinBox, adaptive=True, parameter=self.plugin.parameters['batch_size'])

        self.hyper.addOption('Sequence length:', SpinBox, adaptive=True, parameter=self.plugin.parameters['seqlen'])

        self.optimizer = self.hyper.addOption(None, AdvancedButton, text='Optimizer')
        self.optimizer.linkGroupBox(self.hyper.addOption(None, ParametersFormGroupBox, show=False))
        self.loss_function = self.hyper.addOption(None, AdvancedButton, text='Focal loss')
        self.loss_function.linkGroupBox(self.hyper.addOption(None, ParametersFormGroupBox, show=False))
        self.advanced = self.hyper.addOption(None, AdvancedButton)
        self.advanced.linkGroupBox(self.hyper.addOption(None, ParametersFormGroupBox, show=False))

        self.advanced.group_box.addOption('Random seed:', SpinBox, parameter=self.plugin.parameters['seed'])
        self.optimizer.group_box.addOption('Optimizer:', ComboBox, parameter=self.plugin.parameters['optimizer'])

        self.optimizer.group_box.addOption('Weight decay:', DoubleSpinBox, parameter=self.plugin.parameters['weight_decay'])
        self.optimizer.group_box.addOption('Momentum:', DoubleSpinBox, parameter=self.plugin.parameters['momentum'])

        self.loss_function.group_box.addOption('Gamma:', DoubleSpinBox, single_step=0.1, adaptive=False,
                                               parameter=self.plugin.parameters['focal_gamma'])
        self.loss_function.group_box.addOption('Weight classes:', RadioButton, parameter=self.plugin.parameters['class_weights'])
        self.loss_function.group_box.addOption('L1 regularization:', DoubleSpinBox, single_step=0.01, adaptive=False,
                                               parameter=self.plugin.parameters['L1'])
        self.loss_function.group_box.addOption('L2 regularization:', DoubleSpinBox, single_step=0.01, adaptive=False,
                                               parameter=self.plugin.parameters['L2'])

        self.optimizer.group_box.addOption('Learning rate:', DoubleSpinBox, single_step=1e-5, adaptive=False,
                                           parameter=self.plugin.parameters['learning_rate'])
        self.optimizer.group_box.addOption('Decay rate:', DoubleSpinBox, parameter=self.plugin.parameters['decay_rate'])
        self.optimizer.group_box.addOption('Decay period:', SpinBox, parameter=self.plugin.parameters['decay_period'])

        self.advanced.group_box.addOption('Follow metric:', ComboBox,
                                          parameter=self.plugin.parameters['follow_metric'])
        self.advanced.group_box.addOption('Checkpoint metric:', ComboBox,
                                          parameter=self.plugin.parameters['checkpoint_metric'])
        self.advanced.group_box.addOption('Log metrics:', RadioButton,
                                          parameter=self.plugin.parameters['log_metrics'])

        # self.advanced.group_box.addOption('Early stopping:', RadioButton,
        #                                   parameter=self.plugin.parameters['early_stopping'])
        self.advanced.group_box.addOption('Data augmentation:', RadioButton,
                                          parameter=self.plugin.parameters['augmentation'])
        self.datasets = self.addGroupBox('Datasets')
        self.training_data = self.datasets.addOption('Training dataset:', DoubleSpinBox,
                                                     parameter=self.plugin.parameters['num_training'])
        self.validation_data = self.datasets.addOption('Validation dataset:', DoubleSpinBox,
                                                       parameter=self.plugin.parameters['num_validation'])
        self.test_data = self.datasets.addOption('Test dataset:', DoubleSpinBox, enabled=False,
                                                 parameter=self.plugin.parameters['num_test'])
        self.datasets.addOption('Random seed:', SpinBox, parameter=self.plugin.parameters['dataset_seed'])

        self.preprocessing = self.addGroupBox('Other options')
        self.channels = self.preprocessing.addOption(None, AdvancedButton, text='Preprocessing')
        self.channels.linkGroupBox(self.preprocessing.addOption(None, ParametersFormGroupBox, show=False))

        self.channels.group_box.addOption('Red', ComboBox, parameter=self.plugin.parameters['red_channel'])
        self.channels.group_box.addOption('Green', ComboBox, parameter=self.plugin.parameters['green_channel'])
        self.channels.group_box.addOption('Blue', ComboBox, parameter=self.plugin.parameters['blue_channel'])

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.classifier_selection, self.hyper, self.datasets, self.preprocessing, self.button_box])

        set_connections({self.button_box.accepted    : self.wait_for_training,
                         self.button_box.rejected    : self.close,
                         self.training_data.changed  : lambda _: self.update_datasets(self.training_data, 'num_training'),
                         self.validation_data.changed: lambda _: self.update_datasets(self.validation_data, 'num_validation'),
                         # self.optimizer.changed: self.update_optimizer_options,
                         # PyDetecDiv.app.project_selected: self.update_all,
                         })

        self.plugin.update_parameters('training')

        self.fit_to_contents()
        self.exec()

    def update_datasets(self, changed_dataset: DoubleSpinBox = None, name: str = None) -> None:
        """
        Update the proportion of data to dispatch in training, validation and test datasets. The total must sum to 1 and
        the modifications are constrained to ensure it is the case.

        :param changed_dataset: the dataset that has just been changed
        :param name: the name of the parameter that was changed
        """
        if changed_dataset:
            self.plugin.parameters[name].set_value(changed_dataset.value())
            self.plugin.parameters['num_test'].set_value(
                    1.0 - (self.plugin.parameters['num_training'].value + self.plugin.parameters['num_validation'].value))
            total = self.plugin.parameters['num_training'].value + self.plugin.parameters['num_validation'].value + \
                    self.plugin.parameters['num_test'].value
            if total > 1.0:
                changed_dataset.setValue(changed_dataset.value() - total + 1.0)
        else:
            self.plugin.parameters['num_test'].set_value(
                    1.0 - self.plugin.parameters['num_training'].value - self.plugin.parameters['num_validation'].value)

    def wait_for_training(self) -> None:
        """
        Open a wainting dialog window to wait for completion of training job
        """
        wait_dialog = StdoutWaitDialog('**Training model**', self)
        wait_dialog.resize(500, 300)
        self.job_finished.connect(wait_dialog.stop_redirection)
        self.job_finished.connect(plot_training_results)
        wait_dialog.wait_for(self.run_training)
        self.close()

    def run_training(self) -> None:
        """
        Run a model training job
        """
        self.job_finished.emit(self.plugin.train_model())


class FineTuningDialog(Dialog):
    """
    Dialog window to specify parameters for fine tuning a pretrained model
    """
    job_finished: Signal = Signal(object)

    def __init__(self, plugin: pydetecdiv.plugins.Plugin, title: str = None):
        super().__init__(plugin, title='Fine tuning classification model')

        self.classifier_selection = self.addGroupBox('Classifier')
        self.weights_choice = self.classifier_selection.addOption('Weights', ComboBox,
                                                                  parameter=self.plugin.parameters['weights'])
        self.classifier_selection.addOption('Network', ComboBox, parameter=self.plugin.parameters['model'],
                                            enabled=False)
        self.classifier_selection.addOption('Classes', ComboBox, parameter=self.plugin.parameters['class_names'],
                                            enabled=False)

        self.hyper = self.addGroupBox('Hyper parameters')
        self.hyper.addOption('Epochs:', SpinBox, adaptive=True, parameter=self.plugin.parameters['epochs'])

        self.hyper.addOption('Batch size:', SpinBox, adaptive=True, parameter=self.plugin.parameters['batch_size'])

        self.hyper.addOption('Sequence length:', SpinBox, adaptive=True, parameter=self.plugin.parameters['seqlen'])

        self.optimizer = self.hyper.addOption(None, AdvancedButton, text='Optimizer')
        self.optimizer.linkGroupBox(self.hyper.addOption(None, ParametersFormGroupBox, show=False))
        self.loss_function = self.hyper.addOption(None, AdvancedButton, text='Focal loss')
        self.loss_function.linkGroupBox(self.hyper.addOption(None, ParametersFormGroupBox, show=False))
        self.advanced = self.hyper.addOption(None, AdvancedButton)
        self.advanced.linkGroupBox(self.hyper.addOption(None, ParametersFormGroupBox, show=False))

        self.advanced.group_box.addOption('Random seed:', SpinBox, parameter=self.plugin.parameters['seed'])
        self.optimizer.group_box.addOption('Optimizer:', ComboBox, parameter=self.plugin.parameters['optimizer'])
        self.optimizer.group_box.addOption('Weight decay:', DoubleSpinBox, parameter=self.plugin.parameters['weight_decay'])
        self.optimizer.group_box.addOption('Momentum:', DoubleSpinBox, parameter=self.plugin.parameters['momentum'])

        self.loss_function.group_box.addOption('Gamma:', DoubleSpinBox, single_step=0.1, adaptive=False,
                                               parameter=self.plugin.parameters['focal_gamma'])
        self.loss_function.group_box.addOption('Weight classes:', RadioButton, parameter=self.plugin.parameters['class_weights'])
        self.loss_function.group_box.addOption('L1 regularization:', DoubleSpinBox, parameter=self.plugin.parameters['L1'])
        self.loss_function.group_box.addOption('L2 regularization:', DoubleSpinBox, parameter=self.plugin.parameters['L2'])

        self.optimizer.group_box.addOption('Learning rate:', DoubleSpinBox, decimals=4, single_step=0.01, adaptive=True,
                                           parameter=self.plugin.parameters['learning_rate'])
        self.optimizer.group_box.addOption('Decay rate:', DoubleSpinBox, parameter=self.plugin.parameters['decay_rate'])
        self.optimizer.group_box.addOption('Decay period:', SpinBox, parameter=self.plugin.parameters['decay_period'])

        self.advanced.group_box.addOption('Follow metric:', ComboBox,
                                          parameter=self.plugin.parameters['follow_metric'])
        self.advanced.group_box.addOption('Checkpoint metric:', ComboBox,
                                          parameter=self.plugin.parameters['checkpoint_metric'])

        # self.advanced.group_box.addOption('Early stopping:', RadioButton,
        #                                   parameter=self.plugin.parameters['early_stopping'])
        self.advanced.group_box.addOption('Data augmentation:', RadioButton,
                                          parameter=self.plugin.parameters['augmentation'])

        self.datasets = self.addGroupBox('Datasets')
        self.training_data = self.datasets.addOption('Training dataset:', DoubleSpinBox,
                                                     parameter=self.plugin.parameters['num_training'], enabled=False)
        self.validation_data = self.datasets.addOption('Validation dataset:', DoubleSpinBox,
                                                       parameter=self.plugin.parameters['num_validation'],
                                                       enabled=False)
        self.test_data = self.datasets.addOption('Test dataset:', DoubleSpinBox, enabled=False,
                                                 parameter=self.plugin.parameters['num_test'])
        self.datasets.addOption('Random seed:', Label, parameter=self.plugin.parameters['dataset_seed'])

        self.preprocessing = self.addGroupBox('Other options')
        self.channels = self.preprocessing.addOption(None, AdvancedButton, text='Preprocessing')
        self.channels.linkGroupBox(self.preprocessing.addOption(None, ParametersFormGroupBox, show=False))

        self.channels.group_box.addOption('Red', ComboBox, parameter=self.plugin.parameters['red_channel'],
                                          enabled=False)
        self.channels.group_box.addOption('Green', ComboBox, parameter=self.plugin.parameters['green_channel'],
                                          enabled=False)
        self.channels.group_box.addOption('Blue', ComboBox, parameter=self.plugin.parameters['blue_channel'],
                                          enabled=False)

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.classifier_selection, self.hyper, self.datasets, self.preprocessing, self.button_box])

        set_connections({self.button_box.accepted   : self.wait_for_finetuning,
                         self.button_box.rejected   : self.close,
                         self.weights_choice.changed: self.plugin.select_saved_parameters,
                         })

        self.plugin.update_parameters(groups='finetune')
        self.plugin.select_saved_parameters(self.plugin.parameters['weights'].key)

        self.fit_to_contents()
        self.exec()

    def wait_for_finetuning(self) -> None:
        """
        Open a waiting dialog window to wait for the completion of the fine tuning job
        """
        wait_dialog = StdoutWaitDialog('**Fine-tuning model**', self)
        wait_dialog.resize(500, 300)
        self.job_finished.connect(wait_dialog.stop_redirection)
        self.job_finished.connect(plot_training_results)
        wait_dialog.wait_for(self.run_finetuning)
        self.close()

    def run_finetuning(self) -> None:
        """
        Run fine tuning job
        """
        self.job_finished.emit(self.plugin.train_model(fine_tuning=True))


def plot_training_results(results: tuple[ClassifierTrainingStats, dict[str, ROIDataset], torch.nn.Module, torch.device]) -> None:
    """
    Plots training results (history, confusion matrix, ...)

    :param results: the results from training process
    """
    (train_stats, _, model, _) = results
    module_name, history = train_stats.model_name, train_stats.history
    tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / {module_name}')
    tab.project_name = PyDetecDiv.project_name
    history_plot = plot_history(history)
    tab.addTab(history_plot, 'Training history')
    tab.setCurrentWidget(history_plot)

    interactive = plot_interactive_history(train_stats)
    tab.addTab(interactive, 'Interactive history')

    del model
    torch.cuda.empty_cache()
    gc.collect()


def plot_interactive_history(train_stats: ClassifierTrainingStats) -> QFrame:
    """
    Create a QFrame to display the interactive history in: the left panel shows the training and validation confusion matrices at
    a given epoch, and the right panel shows the metrics history with a movable vertical line used to select the epoch to display
    the confusion matrices for.

    :param train_stats: the training statistics
    :return: the QFrame
    """
    history = train_stats.history

    frame = QFrame()
    layout = QGridLayout()
    frame.setLayout(layout)

    chart_view, epoch_line = plot_metrics_history(train_stats)

    confusion_matrices_view = MatplotViewer(columns=2, rows=2, toolbar=False)
    update_heat_maps(confusion_matrices_view, train_stats, epoch=history.best_epoch)

    layout.addWidget(confusion_matrices_view, 0, 0)
    layout.addWidget(chart_view, 0, 1)

    epoch_line.sigPositionChangeFinished.connect(
        lambda x: update_heat_maps(confusion_matrices_view, train_stats, epoch=int(x.getXPos() + 0.5)))
    epoch_line.sigPositionChangeFinished.connect(lambda x: x.setPos(float(int(x.getXPos() + 0.5))))

    return frame


def update_heat_maps(matplot_view, train_stats, epoch=None):
    """
    Plots the heatmaps in matplot_view for the requested epoch, based on history in train_stats

    :param matplot_view: the MatplotViewer
    :param train_stats: the training statistics
    :param epoch: the requested epoch
    """
    if epoch is None:
        epoch = train_stats.history.best_epoch
    history = train_stats.history
    class_names = train_stats.class_names

    ax = matplot_view.axes

    plot_heatmap(ax[0][0], history.metric_history('ConfusionMatrix_precision')[epoch].numpy(),
                 class_names=class_names, title='precision (train)')
    plot_heatmap(ax[0][1], history.metric_history('ConfusionMatrix_recall')[epoch].numpy(),
                 class_names=class_names, title='recall (train)')
    plot_heatmap(ax[1][0], history.val_metric_history('ConfusionMatrix_precision')[epoch].numpy(),
                 class_names=class_names, title='precision (val)')
    plot_heatmap(ax[1][1], history.val_metric_history('ConfusionMatrix_recall')[epoch].numpy(),
                 class_names=class_names, title='recall (val)')

    matplot_view.show()


def plot_heatmap(axis: matplotlib.axes.Axes, data: np.ndarray, class_names: list[str] = None, title: str | None = None):
    """
    Plot a heatmap in matplotlib axis for classes named in class_names and defined by data

    :param axis: the axis to plot the heatmap in
    :param data: the data defining the heatmap
    :param class_names: the class names
    :param title: the title for the plot
    """
    axis.clear()
    axis.set_title(title, size=10)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right", rotation_mode="anchor", size=8)
    axis.set_yticks(range(len(class_names)), labels=class_names, size=8)
    axis.set_xlabel("Predicted classes", size=8)
    axis.set_ylabel("True classes", size=8)

    im = axis.imshow(data)
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            text = axis.text(j, i, f'{data[i, j].item():0.2f}', ha="center", va="center", color="w", size=8)


def plot_metrics_history(train_stats: ClassifierTrainingStats, epoch: int | None = None) -> tuple[ChartView, pg.InfiniteLine]:
    """
    Creates a ChartView to display the interactive metrics history

    :param train_stats: the training statistics
    :param epoch: the epoch number to initially place the line marking the current epoch
    :return: the ChartView and the epoch_line objects
    """
    history = train_stats.history
    epoch = train_stats.history.best_epoch if epoch is None else epoch

    chart_view = ChartView()
    chart_view.ci.layout.setSpacing(0)
    chart_view.ci.setContentsMargins(0, 0, 0, 0)

    chart_view.addPlot(0, 0, title='Metrics history')
    chart_view.chart(0, 0).addLegend(offset=(-1, -1), anchor=(0, 0), pen=pg.mkPen('k', width=1),
                                     brush=pg.mkBrush('w'))

    i = 0
    for metric_name in train_stats.metrics.keys():
        # if len(history.metric_history(metric_name)[0].shape) <= 1:
        if is_single_value_metric(metric_name):
            chart_view.addLinePlot(history.metric_history(metric_name), row=0, col=0,
                                   pen=pg.mkPen(Colours.palette[i], width=2), name=metric_name)
            chart_view.addLinePlot(history.val_metric_history(metric_name), row=0, col=0,
                                   pen=pg.mkPen(Colours.palette[i], width=2, style=Qt.DashLine))
            scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(Colours.palette[i]))
            best_epoch, best_val = train_stats.best_value(metric_name)
            scatter.addPoints([best_epoch], [best_val])
            chart_view.chart(0, 0).addItem(scatter)
            i += 1
    ticks = [(float(idx), str(idx + 1)) for idx in range(history.num_epochs)]
    chart_view.chart(0, 0).getAxis('bottom').setTicks([ticks, []])

    epoch_line = chart_view.chart(0, 0).addLine(x=epoch, movable=True, pen=pg.mkPen('g', width=3))
    epoch_line.setBounds((0, history.num_epochs - 1))
    return chart_view, epoch_line


def plot_metrics(train_stats: ClassifierTrainingStats) -> MatplotViewer:
    """
    Creates a MatplotViewer to display the metrics history.

    :param train_stats: TrainStats object
    :return: MatplotViewer
    """
    plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, layout='constrained', columns=1, rows=1)
    axs = plot_viewer.axes
    history = train_stats.history
    for metric_name, _ in train_stats.metrics.items():
        if len(history.metric_history(metric_name)[0].shape) <= 1:
            train_line = axs.plot(history.metric_history(metric_name), label=f'train {metric_name}')
            axs.plot(history.val_metric_history(metric_name), label=f'val {metric_name}', color=train_line[0].get_color(),
                     linestyle='dashed')
    plot_viewer.figure.legend(loc='outside right lower')

    plot_viewer.show()
    return plot_viewer


def plot_metric(metric_name, values):
    """
    Creates a MatplotViewer to display the history of the requested metric or loss

    :param metric_name: the metric name
    :param values: the list of values of the requested metric for all epochs
    :return:
    """
    plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, layout='constrained', columns=1, rows=1)
    axs = plot_viewer.axes
    axs.plot(values['train'])
    axs.plot(values['val'])
    axs.set_ylabel(metric_name)
    axs.set_xlabel('epoch')
    plot_viewer.figure.legend(['train', 'val'], loc='outside right lower')

    plot_viewer.show()
    return plot_viewer


def plot_history(history: TrainingHistory) -> MatplotViewer:
    """
    Plots metrics history.

    :param history: metrics history to plot
    :param evaluation: metrics from model evaluation on test dataset, shown as horizontal dashed lines on the plots
    """
    plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=2, rows=1)
    axs = plot_viewer.axes
    history.plot_metric(axs[0], history.main_metric)
    axs[0].axhline(history.val_metric_history(history.main_metric)[history.best_epoch], color='red', linestyle='--')
    axs[0].axvline(history.best_epoch, color='red', linestyle='--')
    history.plot(axs[1])
    axs[1].axhline(min(history.val_loss), color='red', linestyle='--')
    axs[1].axvline(history.best_epoch, color='red', linestyle='--')

    plot_viewer.show()
    return plot_viewer


def plot_confusion_matrix(train_stats: ClassifierTrainingStats, epoch: int = -1, val: bool = False) -> MatplotViewer:
    """
    Creates a MatplotViewer to display training or validation confusion matrices for precision and recall.

    :param train_stats: the training statistcs
    :param epoch: the epoch to plot
    :param val: True is validation matrix is requested
    :return: the MatplotViewer
    """
    plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=2, rows=1)
    plot_viewer.axes[0].set_title('Normalized by row (recall)')
    if val:
        train_stats.val_metrics['ConfusionMatrix_recall'].plot(val=train_stats.val_metrics_values[epoch]['ConfusionMatrix_recall'],
                                                               labels=train_stats.class_names, ax=plot_viewer.axes[0])
    else:
        train_stats.metrics['ConfusionMatrix_recall'].plot(val=train_stats.metrics_values[epoch]['ConfusionMatrix_recall'],
                                                           labels=train_stats.class_names, ax=plot_viewer.axes[0])
    plot_viewer.axes[1].set_title('Normalized by column (precision)')
    if val:
        train_stats.val_metrics['ConfusionMatrix_precision'].plot(
            val=train_stats.val_metrics_values[epoch]['ConfusionMatrix_precision'], labels=train_stats.class_names,
            ax=plot_viewer.axes[1])
    else:
        train_stats.metrics['ConfusionMatrix_precision'].plot(val=train_stats.metrics_values[epoch]['ConfusionMatrix_precision'],
                                                              labels=train_stats.class_names, ax=plot_viewer.axes[1])
    return plot_viewer


def plot_images(dataset: torch.utils.data.Dataset, n: int, class_names: list[str],
                model: torch.nn.Module, device: torch.device) -> MatplotViewer:
    """
    Displays a random selection of images from a dataset along with their ground truth and predictions

    :param dataset: the dataset
    :param n: the number of images to display
    :param class_names: the class names
    :param model: the model
    :param device: the device
    :return: the plot viewer
    """
    plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=n, rows=1)
    for i in range(n):
        idx = random.randint(0, len(dataset) - 1)
        img, target = dataset[idx]
        roi_id = dataset.get_roi_id(idx)
        frame = dataset.get_frame(idx)
        if img.dim() == 4:
            # img = img[math.ceil(img.shape[0] / 2.0)]
            t = random.randint(0, len(target) - 1)
            prediction = model(torch.unsqueeze(img, dim=0).to(device)).argmax(dim=-1).squeeze()[t]
            img = img[t]
            target = target[t]
        else:
            prediction = model(torch.unsqueeze(img, dim=0).to(device)).argmax(dim=-1).squeeze()
            t = 0
        img_channel_last = img.permute([1, 2, 0])
        plot_viewer.axes[i].imshow(img_channel_last.to(torch.float32))
        plot_viewer.axes[i].set_title(f'{class_names[target.item()]} ({class_names[prediction.item()]})')
        plot_viewer.axes[i].set_xlabel(f'{roi_id} [{frame + t}]')
    return plot_viewer


class ImportClassifierDialog(Dialog):
    """
    Import classifier Dialog window
    """
    job_finished: Signal = Signal(object)

    def __init__(self, plugin: pydetecdiv.plugins.Plugin, title: str = None):
        super().__init__(plugin, title='Import classifier from another project')

        classifier_selection = self.addGroupBox('Classifier')
        classifier_selection.addOption(None, ComboBox, parameter=self.plugin.classifiers)

        button_box = self.addButtonBox()

        self.arrangeWidgets([classifier_selection, button_box])

        set_connections({button_box.accepted: self.wait_for_import_classifier,
                         button_box.rejected: self.close,
                         })

        self.fit_to_contents()
        self.exec()

    def wait_for_import_classifier(self) -> None:
        """
        Wait until classifier has been imported
        """
        wait_dialog = StdoutWaitDialog('**Importing classifier**', self)
        wait_dialog.resize(500, 100)
        self.job_finished.connect(wait_dialog.stop_redirection)
        wait_dialog.wait_for(self.run_import_classifier)
        self.close()

    def run_import_classifier(self) -> None:
        """
        Launch classifier import procedure
        """
        self.plugin.import_classifier()
        self.job_finished.emit(True)
