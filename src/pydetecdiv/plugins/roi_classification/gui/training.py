"""
GUI for training and fine tuning models
"""
import gc
import random
import sys

import numpy as np
import torch
from PySide6.QtCore import Signal
from sklearn.metrics import ConfusionMatrixDisplay

import pydetecdiv.plugins
from pydetecdiv.app import StdoutWaitDialog, PyDetecDiv
from pydetecdiv.app.gui.core.widgets.viewers.plots import MatplotViewer

from pydetecdiv.plugins.gui import (ComboBox, AdvancedButton, SpinBox, ParametersFormGroupBox, DoubleSpinBox,
                                    RadioButton, set_connections, Label, Dialog)
from pydetecdiv.plugins.roi_classification.data import ROIDataset
from pydetecdiv.torch import ClassifierTrainingStats


class TuneHyperparamDialog(Dialog):
    """
    A Dialog window to specify hyperparameters for training a model
    """
    job_finished: Signal = Signal(object)

    def __init__(self, plugin: pydetecdiv.plugins.Plugin, title: str = None):
        super().__init__(plugin, title='Tune hyperparameters for classification model')
        self.button_box = self.addButtonBox()
        self.arrangeWidgets([self.button_box])

        set_connections({self.button_box.accepted    : self.wait_for_tunning,
                         self.button_box.rejected    : self.close
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
        self.job_finished.connect(self.show_best_hyperparameters)
        wait_dialog.wait_for(self.run_tuning)
        self.close()

    def run_tuning(self) -> None:
        """
        Run a model training job
        """
        print('Run_tuning method', file=sys.stderr)
        self.job_finished.emit(self.plugin.tune_hyperparameters())

    def show_best_hyperparameters(self, trial):
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

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


def plot_training_results(results: tuple[ClassifierTrainingStats, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, ROIDataset], torch.nn.Module, torch.device]) -> None:
    """
    Plots training results (history, confusion matrix, ...)

    :param results: the results from training process
    """
    (train_stats, ground_truth, predictions, best_gt, best_predictions, dataset, model, device) = results
    module_name, class_names, history, evaluation = train_stats.model_name, train_stats.class_names, train_stats.history, train_stats.evaluation
    tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / {module_name}')
    tab.project_name = PyDetecDiv.project_name
    history_plot = plot_history(history, evaluation)
    tab.addTab(history_plot, 'Training')
    tab.setCurrentWidget(history_plot)

    confusion_matrix_plot = plot_confusion_matrix(ground_truth.cpu(), predictions.cpu(), class_names)
    tab.addTab(confusion_matrix_plot, 'Confusion matrix (last epoch)')

    confusion_matrix_plot = plot_confusion_matrix(best_gt.cpu(), best_predictions.cpu(), class_names)
    tab.addTab(confusion_matrix_plot, 'Confusion matrix (best checkpoint)')

    tab.addTab(plot_images(dataset['train'], 6, class_names, model, device), 'training images')
    tab.addTab(plot_images(dataset['val'], 6, class_names, model, device), 'validation images')
    tab.addTab(plot_images(dataset['test'], 6, class_names, model, device), 'test images')

    del model
    torch.cuda.empty_cache()
    gc.collect()


def plot_history(history: dict, evaluation: dict) -> MatplotViewer:
    """
    Plots metrics history.

    :param history: metrics history to plot
    :param evaluation: metrics from model evaluation on test dataset, shown as horizontal dashed lines on the plots
    """
    plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=2, rows=1)
    axs = plot_viewer.axes
    history.plot(axs[0], 'metric')
    axs[0].axhline(evaluation['metric'].cpu(), color='red', linestyle='--')
    history.plot(axs[1], 'loss')
    axs[1].axhline(evaluation['loss'], color='red', linestyle='--')
    # axs[0].plot(history['train metric'])
    # axs[0].plot(history['val metric'])
    # axs[0].axhline(evaluation['metric'].cpu(), color='red', linestyle='--')
    # axs[0].set_ylabel('metric')
    # axs[0].set_xlabel('epoch')
    # axs[0].legend(['train', 'val'], loc='lower right')
    # axs[1].plot(history['train loss'])
    # axs[1].plot(history['val loss'])
    # axs[1].axhline(evaluation['loss'], color='red', linestyle='--')
    # axs[1].legend(['train', 'val'], loc='upper right')
    # axs[1].set_ylabel('loss')

    plot_viewer.show()
    return plot_viewer


def plot_confusion_matrix(ground_truth: list, predictions: list, class_names: list[str]) -> MatplotViewer:
    """
    Plot the confusion matrix normalized i) by rows (recall in diagonals) and ii) by columns (precision in diagonals)

    :param ground_truth: the ground truth index values
    :param predictions: the predicted index values
    :param class_names: the class names
    :return: the plot viewer where the confusion matrix is plotted
    """
    plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=2, rows=1)
    plot_viewer.axes[0].set_title('Normalized by row (recall)')
    ConfusionMatrixDisplay.from_predictions(ground_truth, predictions, labels=list(range(len(class_names))),
                                            display_labels=class_names, normalize='true', ax=plot_viewer.axes[0], colorbar=False)
    plot_viewer.axes[1].set_title('Normalized by column (precision)')
    ConfusionMatrixDisplay.from_predictions(ground_truth, predictions, labels=list(range(len(class_names))),
                                            display_labels=class_names, normalize='pred', ax=plot_viewer.axes[1], colorbar=False)
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
