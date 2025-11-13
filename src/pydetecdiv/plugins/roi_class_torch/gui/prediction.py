from PySide6.QtCore import Signal

from pydetecdiv.app import StdoutWaitDialog
from pydetecdiv.plugins.gui import (ComboBox, set_connections, SpinBox, AdvancedButton, ParametersFormGroupBox,
                                    ListWidget, Dialog)


class PredictionDialog(Dialog):
    finished = Signal(bool)

    def __init__(self, plugin, title=None):
        super().__init__(plugin, title='Predict classes')

        self.classifier_selection = self.addGroupBox('Classifier')
        self.weights_choice = self.classifier_selection.addOption('Weights', ComboBox, parameter=self.plugin.parameters['weights'])
        self.classifier_selection.addOption('Network', ComboBox, parameter=self.plugin.parameters['model'], enabled=False)
        self.classifier_selection.addOption('Classes', ComboBox, parameter=self.plugin.parameters['class_names'], enabled=False)

        # self.fov_selection = self.addGroupBox('Select FOVs')
        # self.fov_selection.addOption(None, ListWidget, parameter=self.plugin.parameters['fov'])

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

        self.arrangeWidgets([self.classifier_selection, #self.fov_selection,
                             self.hyper, self.preprocessing, self.button_box])

        set_connections({self.button_box.accepted: self.wait_for_prediction,
                         self.button_box.rejected: self.close,
                         self.weights_choice.changed: self.plugin.select_saved_parameters,
                         })
        #
        self.plugin.update_parameters(groups='prediction')
        self.plugin.select_saved_parameters(self.plugin.parameters['weights'].key)

        self.fit_to_contents()
        self.exec()

    def wait_for_prediction(self):
        wait_dialog = StdoutWaitDialog('**Starting prediction run**', self)
        wait_dialog.resize(500, 300)
        self.finished.connect(wait_dialog.stop_redirection)
        wait_dialog.wait_for(self.run_prediction)
        self.close()

    def run_prediction(self):
        self.plugin.predict()
        self.finished.emit(True)
