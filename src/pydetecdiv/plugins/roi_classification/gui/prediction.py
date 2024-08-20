from pydetecdiv.plugins import Dialog
from pydetecdiv.plugins.gui import ComboBox, set_connections


class PredictionDialog(Dialog):
    def __init__(self, plugin, title=None):
        super().__init__(plugin, title='Predict classes')

        self.classifier_selection = self.addGroupBox('Classifier')
        self.network = self.classifier_selection.addOption('Network:', ComboBox,
                                                           parameter=plugin.parameters['model'])
        self.weights = self.classifier_selection.addOption('Weights:', ComboBox,
                                                           parameter=plugin.parameters['weights'])
        self.classes = self.classifier_selection.addOption('Classes:', ComboBox,
                                                           parameter=plugin.parameters['class_names'])

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.classifier_selection, self.button_box])

        set_connections({self.button_box.accepted: self.run_prediction,
                         self.button_box.rejected: self.close,
                         self.network.selected: self.plugin.update_model_weights,
                         self.weights.selected: self.plugin.update_class_names,
                         })

        self.fit_to_contents()
        self.exec()

    def run_prediction(self):
        print('running prediction')
