from pydetecdiv.plugins import Dialog

from pydetecdiv.plugins.gui import ComboBox, AdvancedButton, SpinBox, ParametersFormGroupBox, DoubleSpinBox, \
    RadioButton, set_connections, Label


class TrainingDialog(Dialog):
    def __init__(self, plugin, title=None):
        super().__init__(plugin, title='Training classification model')

        self.classifier_selection = self.addGroupBox('Classifier')
        self.classifier_selection.addOption('Network', ComboBox, parameter=self.plugin.parameters['model'])
        self.classifier_selection.addOption('Classes', ComboBox, parameter=self.plugin.parameters['class_names'])

        self.hyper = self.addGroupBox('Hyper parameters')
        self.hyper.addOption('Epochs:', SpinBox, adaptive=True, parameter=self.plugin.parameters['epochs'])

        self.hyper.addOption('Batch size:', SpinBox, adaptive=True, parameter=self.plugin.parameters['batch_size'])

        self.hyper.addOption('Sequence length:', SpinBox, adaptive=True, parameter=self.plugin.parameters['seqlen'])

        self.advanced = self.hyper.addOption(None, AdvancedButton)
        self.advanced.linkGroupBox(self.hyper.addOption(None, ParametersFormGroupBox, show=False))
        self.advanced.group_box.addOption('Random seed:', SpinBox, parameter=self.plugin.parameters['seed'])
        self.advanced.group_box.addOption('Optimizer:', ComboBox, parameter=self.plugin.parameters['optimizer'])

        self.advanced.group_box.addOption('Learning rate:', DoubleSpinBox, decimals=4, single_step=0.01, adaptive=True,
                                          parameter=self.plugin.parameters['learning_rate'])
        self.advanced.group_box.addOption('Decay rate:', DoubleSpinBox, parameter=self.plugin.parameters['decay_rate'])
        self.advanced.group_box.addOption('Decay period:', SpinBox, parameter=self.plugin.parameters['decay_period'])
        self.advanced.group_box.addOption('Momentum:', DoubleSpinBox, parameter=self.plugin.parameters['momentum'])
        self.advanced.group_box.addOption('Checkpoint metric:', ComboBox,
                                          parameter=self.plugin.parameters['checkpoint_metric'])

        self.advanced.group_box.addOption('Early stopping:', RadioButton,
                                          parameter=self.plugin.parameters['early_stopping'])

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

        set_connections({self.button_box.accepted: self.plugin.train_model,
                         self.button_box.rejected: self.close,
                         self.training_data.changed: lambda _: self.update_datasets(self.training_data),
                         self.validation_data.changed: lambda _: self.update_datasets(self.validation_data),
                         # self.optimizer.changed: self.update_optimizer_options,
                         # PyDetecDiv.app.project_selected: self.update_all,
                         })

        self.plugin.update_parameters('training')

        self.fit_to_contents()
        self.exec()

    def update_datasets(self, changed_dataset=None):
        """
        Update the proportion of data to dispatch in training, validation and test datasets. The total must sum to 1 and
        the modifications are constrained to ensure it is the case.

        :param changed_dataset: the dataset that has just been changed
        """
        if changed_dataset:
            self.plugin.parameters['num_test'].set_value(1.0 - (self.plugin.parameters['num_training'].value + self.plugin.parameters['num_validation'].value))
            total = self.plugin.parameters['num_training'].value + self.plugin.parameters['num_validation'].value + self.plugin.parameters['num_test'].value
            if total > 1.0:
                changed_dataset.setValue(changed_dataset.value() - total + 1.0)
        else:
            self.plugin.parameters['num_test'].set_value(1.0 - self.plugin.parameters['num_training'].value - self.plugin.parameters['num_validation'].value)

class FineTuningDialog(Dialog):
    def __init__(self, plugin, title=None):
        super().__init__(plugin, title='Fine tuning classification model')

        self.classifier_selection = self.addGroupBox('Classifier')
        self.weights_choice = self.classifier_selection.addOption('Weights', ComboBox, parameter=self.plugin.parameters['weights'])
        self.classifier_selection.addOption('Network', ComboBox, parameter=self.plugin.parameters['model'], enabled=False)
        self.classifier_selection.addOption('Classes', ComboBox, parameter=self.plugin.parameters['class_names'], enabled=False)

        self.hyper = self.addGroupBox('Hyper parameters')
        self.hyper.addOption('Epochs:', SpinBox, adaptive=True, parameter=self.plugin.parameters['epochs'])

        self.hyper.addOption('Batch size:', SpinBox, adaptive=True, parameter=self.plugin.parameters['batch_size'])

        self.hyper.addOption('Sequence length:', SpinBox, adaptive=True, parameter=self.plugin.parameters['seqlen'])

        self.advanced = self.hyper.addOption(None, AdvancedButton)
        self.advanced.linkGroupBox(self.hyper.addOption(None, ParametersFormGroupBox, show=False))
        self.advanced.group_box.addOption('Random seed:', SpinBox, parameter=self.plugin.parameters['seed'])
        self.advanced.group_box.addOption('Optimizer:', ComboBox, parameter=self.plugin.parameters['optimizer'])

        self.advanced.group_box.addOption('Learning rate:', DoubleSpinBox, decimals=4, single_step=0.01, adaptive=True,
                                          parameter=self.plugin.parameters['learning_rate'])
        self.advanced.group_box.addOption('Decay rate:', DoubleSpinBox, parameter=self.plugin.parameters['decay_rate'])
        self.advanced.group_box.addOption('Decay period:', SpinBox, parameter=self.plugin.parameters['decay_period'])
        self.advanced.group_box.addOption('Momentum:', DoubleSpinBox, parameter=self.plugin.parameters['momentum'])
        self.advanced.group_box.addOption('Checkpoint metric:', ComboBox,
                                          parameter=self.plugin.parameters['checkpoint_metric'])

        self.advanced.group_box.addOption('Early stopping:', RadioButton,
                                          parameter=self.plugin.parameters['early_stopping'])

        self.datasets = self.addGroupBox('Datasets')
        self.training_data = self.datasets.addOption('Training dataset:', DoubleSpinBox,
                                                     parameter=self.plugin.parameters['num_training'], enabled=False)
        self.validation_data = self.datasets.addOption('Validation dataset:', DoubleSpinBox,
                                                       parameter=self.plugin.parameters['num_validation'], enabled=False)
        self.test_data = self.datasets.addOption('Test dataset:', DoubleSpinBox, enabled=False,
                                                 parameter=self.plugin.parameters['num_test'])
        self.datasets.addOption('Random seed:', Label, parameter=self.plugin.parameters['dataset_seed'])

        self.preprocessing = self.addGroupBox('Other options')
        self.channels = self.preprocessing.addOption(None, AdvancedButton, text='Preprocessing')
        self.channels.linkGroupBox(self.preprocessing.addOption(None, ParametersFormGroupBox, show=False))

        self.channels.group_box.addOption('Red', ComboBox, parameter=self.plugin.parameters['red_channel'], enabled=False)
        self.channels.group_box.addOption('Green', ComboBox, parameter=self.plugin.parameters['green_channel'], enabled=False)
        self.channels.group_box.addOption('Blue', ComboBox, parameter=self.plugin.parameters['blue_channel'], enabled=False)

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.classifier_selection, self.hyper, self.datasets, self.preprocessing, self.button_box])

        set_connections({self.button_box.accepted: self.plugin.train_model,
                         self.button_box.rejected: self.close,
                         self.weights_choice.changed: self.plugin.select_saved_parameters,
                         })
        #
        self.plugin.update_parameters(groups='finetune')
        self.plugin.select_saved_parameters(self.plugin.parameters['weights'].key)

        self.fit_to_contents()
        self.exec()

class ImportClassifierDialog(Dialog):
    def __init__(self, plugin, title=None):
        super().__init__(plugin, title='Import classifier from another project')

        classifier_selection = self.addGroupBox('Classifier')
        classifier_selection.addOption(None, ComboBox, parameter=self.plugin.classifiers)

        button_box = self.addButtonBox()

        self.arrangeWidgets([classifier_selection, button_box])

        set_connections({button_box.accepted: self.plugin.import_classifier,
                         button_box.rejected: self.close,
                         })

        self.fit_to_contents()
        self.exec()
