from pydetecdiv.plugins import Dialog
from pydetecdiv.plugins.gui import ComboBox, set_connections, SpinBox, AdvancedButton, ParametersFormGroupBox, \
    DoubleSpinBox, TableView, ListView


class PredictionDialog(Dialog):
    def __init__(self, plugin, title=None):
        super().__init__(plugin, title='Predict classes')

        self.classifier_selection = self.addGroupBox('Classifier')
        self.weights_choice = self.classifier_selection.addOption('Weights', ComboBox, parameter=self.plugin.parameters['weights'])
        self.classifier_selection.addOption('Network', ComboBox, parameter=self.plugin.parameters['model'], enabled=False)
        self.classifier_selection.addOption('Classes', ComboBox, parameter=self.plugin.parameters['class_names'], enabled=False)

        self.fov_selection = self.addGroupBox('Select FOVs')
        self.fov_selection.addOption(None, ListView, parameter=self.plugin.parameters['fov_list'])

        self.hyper = self.addGroupBox('Hyper parameters')
        self.hyper.addOption('Batch size:', SpinBox, adaptive=True, parameter=self.plugin.parameters['batch_size'])

        self.hyper.addOption('Sequence length:', SpinBox, adaptive=True, parameter=self.plugin.parameters['seqlen'])

        self.preprocessing = self.addGroupBox('Other options')
        self.channels = self.preprocessing.addOption(None, AdvancedButton, text='Preprocessing')
        self.channels.linkGroupBox(self.preprocessing.addOption(None, ParametersFormGroupBox, show=False))

        self.channels.group_box.addOption('Red', ComboBox, parameter=self.plugin.parameters['red_channel'], enabled=False)
        self.channels.group_box.addOption('Green', ComboBox, parameter=self.plugin.parameters['green_channel'], enabled=False)
        self.channels.group_box.addOption('Blue', ComboBox, parameter=self.plugin.parameters['blue_channel'], enabled=False)

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.classifier_selection, self.fov_selection,
                             self.hyper, self.preprocessing, self.button_box])

        set_connections({self.button_box.accepted: self.run_prediction,
                         self.button_box.rejected: self.close,
                         self.weights_choice.changed: self.plugin.select_saved_parameters,
                         })
        #
        self.plugin.update_parameters(groups='prediction')
        self.plugin.select_saved_parameters(self.plugin.parameters['weights'].key)

        self.fit_to_contents()
        self.exec()

    def run_prediction(self):
        print('running prediction')
        print(self.plugin.parameters['fov_list'].value)
