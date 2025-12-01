"""
Display model summary
"""
import sys

from PySide6.QtCore import Signal
from torchinfo import summary

import pydetecdiv.plugins
from pydetecdiv.app import StdoutWaitDialog
from pydetecdiv.plugins.gui import (ComboBox, set_connections, SpinBox, Dialog)


class ModelInfoDialog(Dialog):
    """
    A Dialog window to choose a model to display information about
    """
    finished = Signal(bool)

    def __init__(self, plugin: pydetecdiv.plugins.Plugin, title: str = None):
        super().__init__(plugin, title='Show model info')
        self.plugin = plugin

        self.classifier_selection = self.addGroupBox('Classifier')
        self.classifier_selection.addOption('Network', ComboBox, parameter=self.plugin.parameters['model'], enabled=True)
        self.classifier_selection.addOption('Classes', ComboBox, parameter=self.plugin.parameters['class_names'], enabled=True)

        self.hyper = self.addGroupBox('Hyper parameters')
        self.hyper.addOption('Batch size:', SpinBox, adaptive=True, parameter=self.plugin.parameters['batch_size'])
        self.hyper.addOption('Sequence length:', SpinBox, adaptive=True, parameter=self.plugin.parameters['seqlen'])

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.classifier_selection, self.hyper, self.button_box])

        set_connections({self.button_box.accepted: self.run_info,
                         self.button_box.rejected: self.close,
                         })
        #
        self.plugin.update_parameters(groups='info')

        self.fit_to_contents()
        self.exec()

    def wait_for_info(self) -> None:
        """
        Waiting dialog window to wait for information display to complete
        """
        wait_dialog = StdoutWaitDialog(f'**Displaying info about {self.plugin.parameters["model"].key}**', self)
        wait_dialog.resize(500, 300)
        self.finished.connect(wait_dialog.stop_redirection)
        wait_dialog.wait_for(self.run_info)
        self.close()

    def run_info(self) -> None:
        """
        Display model information (summary)
        """
        model, model_name = self.plugin.load_model(pretrained=False)
        print(model_name, file=sys.stderr)
        if len(model.expected_shape) == 4:
            summary(model.cuda(), (self.plugin.parameters['batch_size'].value,) + model.expected_shape[1:4])
        else:
            summary(model.cuda(), (self.plugin.parameters['batch_size'].value,
                                   self.plugin.parameters['seqlen'].value,) + model.expected_shape[2:5])
        self.finished.emit(True)
