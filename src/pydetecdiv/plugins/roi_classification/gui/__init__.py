"""
GUI for deep-learning ROI classification plugin
"""
import json
import os
import sqlalchemy

import keras.optimizers
from PySide6.QtSql import QSqlQuery, QSqlDatabase
from PySide6.QtWidgets import QFileDialog, QDialogButtonBox

from pydetecdiv.settings import get_config_value
from pydetecdiv.utils import Singleton
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, get_project_dir

from pydetecdiv.plugins.gui import Dialog, DialogButtonBox, FormGroupBox, ComboBox, SpinBox, DoubleSpinBox, LineEdit, \
    TableView, set_connections, AdvancedButton, RadioButton
from pydetecdiv.plugins.roi_classification.gui.ImportAnnotatedROIs import FOV2ROIlinks


# @singleton
class ROIclassificationDialog(Dialog, Singleton):
    """
    Dialog window to handle the options Form for ROI classification plugin
    """

    def __init__(self, plugin, title=None):
        super().__init__(plugin, title=title)
        self.plugin.parameter_widgets.add_groups(['training', 'classify', 'annotate', 'create'])

        self.controller = self.addGroupBox('Choose action')
        self.action_menu = self.controller.addOption('Action:', ComboBox,
                                                     items={  # 'Create new model': self.plugin.create_model,
                                                         'Annotate ROIs': self.plugin.annotate_rois,
                                                         'Train model': self.plugin.train_model,
                                                         'Classify ROIs': self.plugin.predict},
                                                     selected='Classify ROIs')

        self.classifier_selection = self.addGroupBox('Classifier')
        self.network = self.classifier_selection.addOption('Network:', ComboBox,
                                                           parameter=(['training', 'classify'], 'model'))
        self.weights = self.classifier_selection.addOption('Weights:', ComboBox,
                                                           parameter=(['training', 'classify'], 'weights'))
        self.classes = self.classifier_selection.addOption('Classes:', ComboBox, parameter=(
            ['training', 'classify', 'annotate'], 'class_names'), editable=True)
        self.training_advanced = self.classifier_selection.addOption(None, AdvancedButton)
        self.training_advanced.linkGroupBox(self.classifier_selection.addOption(None, FormGroupBox, show=False))
        self.weight_seed = self.training_advanced.group_box.addOption('Random seed:', SpinBox, value=42,
                                                                      parameter=(['training'], 'seed'))
        self.optimizer = self.training_advanced.group_box.addOption('Optimizer:', ComboBox,
                                                                    items={'SGD': keras.optimizers.SGD,
                                                                           'Adam': keras.optimizers.Adam,
                                                                           'Adadelta': keras.optimizers.Adadelta,
                                                                           'Adamax': keras.optimizers.Adamax,
                                                                           'Nadam': keras.optimizers.Nadam, },
                                                                    selected='SGD',
                                                                    parameter=(['training'], 'optimizer'))

        self.learning_rate = self.training_advanced.group_box.addOption('Learning rate:', DoubleSpinBox,
                                                                        range=(0.00001, 1.0), decimals=4, value=0.001,
                                                                        parameter=(['training'], 'learning_rate'))
        self.decay_rate = self.training_advanced.group_box.addOption('Decay rate:', DoubleSpinBox, value=0.95,
                                                                     parameter=(['training'], 'decay_rate'))
        self.decay_freq = self.training_advanced.group_box.addOption('Decay period:', SpinBox, value=2,
                                                                     parameter=(['training'], 'decay_period'))
        self.momentum = self.training_advanced.group_box.addOption('Momentum:', DoubleSpinBox, value=0.9,
                                                                   parameter=(['training'], 'momentum'))
        self.checkpoint_metric = self.training_advanced.group_box.addOption('Checkpoint metric:', ComboBox,
                                                                            items={'Loss': 'val_loss',
                                                                                   'Accuracy': 'val_accuracy', },
                                                                            selected='Loss',
                                                                            parameter=(['training'],
                                                                                       'checkpoint_metric'))

        self.early_stopping = self.training_advanced.group_box.addOption('Early stopping:', RadioButton,
                                                                         parameter=(['training'], 'early_stopping'))
        # self.class_weights = self.training_advanced.group_box.addOption('Class weights:', RadioButton,
        #                                                                  parameter=(['training'], 'class_weights'))

        self.roi_selection = self.addGroupBox('Select ROIs')
        self.table = self.roi_selection.addOption(None, TableView, multiselection=True, behavior='rows')
        self.selection_model = self.table.selectionModel()

        self.roi_sample = self.addGroupBox('Sample ROIs')
        self.roi_number = self.roi_sample.addOption('ROI sample size:', SpinBox, adaptive=True,
                                                    parameter=(['annotate'], 'roi_number'))

        self.roi_import = self.addGroupBox('Import annotated ROIs')
        self.roi_import_box = self.roi_import.addOption('Select annotation file:', DialogButtonBox,
                                                        buttons=QDialogButtonBox.Open)

        self.datasets = self.addGroupBox('ROI datasets')
        self.training_data = self.datasets.addOption('Training dataset:', DoubleSpinBox, value=0.6,
                                                     parameter=(['training'], 'num_training'))
        self.validation_data = self.datasets.addOption('Validation dataset:', DoubleSpinBox, value=0.2,
                                                       parameter=(['training'], 'num_validation'))
        self.test_data = self.datasets.addOption('Test dataset:', DoubleSpinBox, value=0.2, enabled=False)
        self.datasets_advanced = self.datasets.addOption(None, AdvancedButton)
        self.datasets_advanced.linkGroupBox(self.datasets.addOption(None, FormGroupBox, show=False))
        self.datasets_seed = self.datasets_advanced.group_box.addOption('Random seed:', SpinBox, value=42,
                                                                        parameter=(['training'], 'dataset_seed'))

        self.preprocessing = self.addGroupBox('Preprocessing')
        self.channels = self.preprocessing.addOption(None, FormGroupBox, title='z to channel')
        # self.channels.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        self.red_channel = self.channels.addOption('Red', ComboBox, parameter=(['training', 'classify'], 'red_channel'))
        self.green_channel = self.channels.addOption('Green', ComboBox,
                                                     parameter=(['training', 'classify'], 'green_channel'))
        self.blue_channel = self.channels.addOption('Blue', ComboBox,
                                                    parameter=(['training', 'classify'], 'blue_channel'))

        self.misc_box = self.addGroupBox('Miscellaneous')

        self.epochs = self.misc_box.addOption('Epochs:', SpinBox, value=16, adaptive=True,
                                              parameter=(['training'], 'epochs'))

        self.batch_size = self.misc_box.addOption('Batch size:', SpinBox, range=(2, 4096), value=128, adaptive=True,
                                                  parameter=(['training', 'classify'], 'batch_size'))

        self.seq_length = self.misc_box.addOption('Sequence length:', SpinBox, value=50, adaptive=True,
                                                  parameter=(['training', 'classify'], 'seqlen'))

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([
            self.controller,
            self.classifier_selection,
            self.preprocessing,
            self.roi_selection,
            self.roi_sample,
            self.roi_import,
            self.datasets,
            self.misc_box,
            self.button_box
        ])

        set_connections({self.button_box.accepted: self.plugin.run,
                         self.button_box.rejected: self.close,
                         self.roi_import_box.accepted: self.import_annotated_rois,
                         # self.network.selected: [self.update_classes, self.update_model_weights],
                         self.network.selected: self.update_model_weights,
                         self.weights.selected: self.update_classes,
                         # self.action_menu.selected: [self.adapt, self.update_classes, self.update_model_weights],
                         self.action_menu.selected: [self.adapt, self.update_model_weights],
                         self.training_data.changed: lambda _: self.update_datasets(self.training_data),
                         self.validation_data.changed: lambda _: self.update_datasets(self.validation_data),
                         self.optimizer.changed: self.update_optimizer_options,
                         PyDetecDiv.app.project_selected: self.update_all,
                         PyDetecDiv.app.saved_rois: self.set_table_view,
                         })

        self.plugin.load_models(self)
        self.update_all()
        self.adapt()

    def update_all(self):
        """
        Update all portions of the options Form GUI
        """
        self.update_datasets()
        self.set_table_view(PyDetecDiv.project_name)
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.update_list(project)
            self.update_sequence_length(project)
            self.update_num_rois(project)

    def toggleAdvanced(self, button):
        """
        Toggle the advanced button between + (expand) and - (collapse) to show or hide advanced options

        :param button:
        """
        button.toggle()
        self.adapt()

    def adapt(self):
        """
        Modify the appearance of the GUI according to the selected action
        """
        match (self.action_menu.currentText()):
            case 'Create new model':
                self.roi_selection.hide()
                self.roi_sample.hide()
                self.roi_import.hide()
                self.classifier_selection.setRowVisible(1, False)
                self.training_advanced.hide()
                self.preprocessing.show()
                self.misc_box.hide()
                self.network.setEditable(True)
                self.classes.setEditable(False)
                self.datasets.hide()
            case 'Annotate ROIs':
                self.roi_selection.hide()
                self.roi_sample.show()
                self.roi_import.show()
                self.classifier_selection.setRowVisible(self.network, False)
                self.classifier_selection.setRowVisible(self.weights, False)
                self.training_advanced.hide()
                self.preprocessing.hide()
                self.misc_box.hide()
                self.network.setEditable(False)
                self.classes.setEditable(True)
                self.datasets.hide()
            case 'Train model':
                self.roi_selection.hide()
                self.roi_sample.hide()
                self.roi_import.hide()
                self.classifier_selection.setRowVisible(self.network, True)
                self.classifier_selection.setRowVisible(self.weights, True)
                self.training_advanced.show()
                self.preprocessing.show()
                self.misc_box.show()
                self.misc_box.setRowVisible(self.epochs, True)
                self.network.setEditable(False)
                self.datasets.show()
                self.classes.setEditable(False)
            case 'Classify ROIs':
                self.roi_selection.show()
                self.roi_sample.hide()
                self.roi_import.hide()
                self.classifier_selection.setRowVisible(self.network, True)
                self.classifier_selection.setRowVisible(self.weights, True)
                self.training_advanced.hide()
                self.preprocessing.show()
                self.misc_box.show()
                self.misc_box.setRowVisible(self.epochs, False)
                self.network.setEditable(False)
                self.datasets.hide()
                self.classes.setEditable(False)
            case _:
                pass
        self.fit_to_contents()

    def set_table_view(self, project_name):
        """
        Set the content of the Table view to display the available ROIs to classify

        :param project_name: the name of the project
        """
        if project_name:
            with pydetecdiv_project(project_name) as project:
                self.update_list(project)

    def update_list(self, project):
        """
        Update the list of FOVs and the number of corresponding ROIs

        :param project: the current project
        """
        db = QSqlDatabase("QSQLITE")
        db.setDatabaseName(project.repository.name)
        db.open()
        query = QSqlQuery(
            "SELECT FOV.name as 'FOV name', count(ROI.id_) as 'ROIs', ImageResource.tdim as 'frames',"
            "ImageResource.zdim as 'layers', ImageResource.cdim as 'channels'"
            " FROM FOV, ImageResource "
            "JOIN ROI ON ROI.fov == FOV.id_ "
            "WHERE FOV.id_ == ImageResource.fov "
            "GROUP BY FOV.id_",
            db=db)
        self.table.setQuery(query)
        self.table.resizeColumnsToContents()
        n_layers = project.get_object('ImageResource', 1).zdim if project.get_object('ImageResource', 1) else 0
        self.red_channel.clear()
        self.green_channel.clear()
        self.blue_channel.clear()
        self.red_channel.addItems([str(i) for i in range(n_layers)])
        self.green_channel.addItems([str(i) for i in range(n_layers)])
        self.blue_channel.addItems([str(i) for i in range(n_layers)])
        self.green_channel.setCurrentIndex(1)
        self.blue_channel.setCurrentIndex(2)

    def update_num_rois(self, project):
        """
        Update the maximum value for image sequence according to the umber of frames in the dataset

        :param project: the current project
        """
        num_rois = project.count_objects('ROI')
        self.roi_number.setRange(1, num_rois)
        self.roi_number.setValue(int(num_rois / 10))

    def update_sequence_length(self, project):
        """
        Update the maximum value for image sequence according to the umber of frames in the dataset

        :param project: the current project
        """
        db = QSqlDatabase("QSQLITE")
        db.setDatabaseName(project.repository.name)
        db.open()
        query = QSqlQuery(
            "SELECT min(tdim) from ImageResource",
            db=db)
        if query.first() and query.record().value('min(tdim)'):
            self.seq_length.setRange(1, query.record().value('min(tdim)'))

    def update_model_weights(self):
        """
        Update the list of model weights associated with the currently selected network
        """
        model_path = self.network.currentData().__path__[0]
        w_files = [os.path.join(model_path, f) for f in os.listdir(model_path)
                   if os.path.isfile(os.path.join(model_path, f)) and f.endswith('.h5')]

        try:
            user_path = os.path.join(get_project_dir(), 'roi_classification', 'models', self.network.currentText())
            w_files.extend([os.path.join(user_path, f) for f in os.listdir(user_path)
                            if os.path.isfile(os.path.join(user_path, f)) and f.endswith('.h5')])
        except FileNotFoundError:
            pass

        self.weights.clear()
        # _ = [self.weights.addItem(os.path.basename(f), userData=f) for f in w_files]
        weights = {os.path.basename(f): f for f in w_files}
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        if self.action_menu.currentText() != 'Classify ROIs':
            self.weights.addItem('None', userData=None)
        elif len(w_files) == 0:
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self.weights.addItemDict(weights)

    def update_classes(self):
        """
        Update the classes associated with the currently selected model
        """
        # self.classes.setText(json.dumps(get_class_names()))
        self.classes.clear()
        if self.weights.currentText():
            self.classes.addItemDict(self.get_class_names(self.weights.currentText()))

    def update_datasets(self, changed_dataset=None):
        """
        Update the proportion of data to dispatch in training, validation and test datasets. The total must sum to 1 and
        the modifications are constrained to ensure it is the case.

        :param changed_dataset: the dataset that has just been changed
        """
        if changed_dataset:
            self.test_data.setValue(1.0 - (self.training_data.value() + self.validation_data.value()))
            total = self.training_data.value() + self.validation_data.value() + self.test_data.value()
            if total > 1.0:
                changed_dataset.setValue(changed_dataset.value() - total + 1.0)
        else:
            self.test_data.setValue(1 - self.training_data.value() - self.validation_data.value())

    def update_optimizer_options(self):
        """
        Adapt the optimizer's available options to the currently selected optimizer
        """
        match self.optimizer.currentText():
            case 'SGD':
                self.training_advanced.group_box.setRowVisible(self.momentum, True)
            case _:
                self.training_advanced.group_box.setRowVisible(self.momentum, False)

    def import_annotated_rois(self):
        """
        Select a csv file containing ROI frames annotations and open a FOV2ROIlinks window to load the data it contains
        into the database as FOVs and ROIs with annotations.
        """
        filters = ["csv (*.csv)", ]
        annotation_file, _ = QFileDialog.getOpenFileName(self, caption='Choose file with annotated ROIs',
                                                         dir='.',
                                                         filter=";;".join(filters),
                                                         selectedFilter=filters[0])
        FOV2ROIlinks(annotation_file, self.plugin)

    @staticmethod
    def get_class_names(weight_file):
        """
        Get the class names for a project

        :return: the list of classes from the last annotation run for this project
        """
        if weight_file != 'None':
            clause = f"(run.command='train_model') AND (best_weights='{weight_file}' OR last_weights='{weight_file}')"
        else:
            clause = (f"annotator='{get_config_value('project', 'user')}' "
                      f"AND (run.command='annotate_rois' OR run.command='import_annotated_rois') ")
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = list(project.repository.session.execute(
                sqlalchemy.text(f"SELECT "
                                f"run.parameters ->> '$.annotator' as annotator, "
                                f"run.parameters ->> '$.class_names' as class_names, "
                                f"run.parameters ->> '$.best_weights' as best_weights, "
                                f"run.parameters ->> '$.last_weights' as last_weights "
                                f"FROM run "
                                f"WHERE {clause} "
                                f"ORDER BY run.id_ DESC;")))
            # class_names = json.loads(results[-1][1])
            class_names = {r[1]: None for r in results}
        return class_names
