import importlib
import json
import os
import pkgutil
import sys
from datetime import datetime

import cv2
import fastremap
import tables as tbl
import numpy as np
import pandas as pd
import sqlalchemy
import torch
from PySide6.QtGui import QAction
from PySide6.QtSql import QSqlDatabase, QSqlQuery
from PySide6.QtWidgets import QMenu, QFileDialog, QMessageBox, QGraphicsRectItem

from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import registry
from sqlalchemy.types import JSON
from torchinfo import summary

from torch import optim

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, get_project_dir, project_list
from pydetecdiv.domain.Run import Run
from pydetecdiv.domain.Project import Project
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.plugins.parameters import ItemParameter, ChoiceParameter, IntParameter, FloatParameter, CheckParameter

from . import models
from .data import prepare_data_for_training
from .gui.ImportAnnotatedROIs import FOV2ROIlinks
from .gui.classification import ManualAnnotator, PredictionViewer, DefineClassesDialog
from .gui.prediction import PredictionDialog
from .gui.training import TrainingDialog, FineTuningDialog, ImportClassifierDialog

from pydetecdiv.settings import get_plugins_dir

Base = registry().generate_base()


class Results(Base):
    """
    The DAO defining and handling the table to store results
    """
    __tablename__ = 'roi_class_torch'
    __table_args__ = {'extend_existing': True}
    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    run = Column(Integer, nullable=False, index=True)
    roi = Column(Integer, nullable=False, index=True)
    t = Column(Integer, nullable=False, index=True)
    predictions = Column(JSON)
    class_name = Column(String)
    score = Column(Float)

    def save(self, project: Project, run: Run, roi: ROI, t: int, predictions: np.ndarray, class_names: list[str]) -> None:
        """
        Save the results from a plugin run on a ROI at time t into the database

        :param project: the current project
        :param run: the current run
        :param roi: the current ROI
        :param t: the current frame
        :param predictions: the list of prediction values
        :param class_names: the class names
        """
        self.roi = roi.id_
        self.run = run.id_
        self.t = t
        self.predictions = predictions.tolist()
        max_score, max_index = max((value, index) for index, value in enumerate(predictions))
        self.class_name = class_names[max_index]
        self.score = max_score
        project.repository.session.add(self)


class TrainingData(Base):
    """
    The DAO defining and handling the table to store information about ROIs in training, validation and test datasets
    """
    __tablename__ = 'roi_class_datasets'
    __table_args__ = {'extend_existing': True}
    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    dataset = Column(Integer, nullable=False, index=True)
    roi = Column(Integer, nullable=False, index=True)
    t = Column(Integer, nullable=False, index=True)
    target = Column(JSON)

    def save(self, project: Project, roi_id: int, t: int, target: int, dataset: int) -> None:
        """
        Save the Training datasets details for reproducibility

        :param project: the current project
        :param roi_id: the current ROI
        :param t: the current frame
        :param target: the target value
        :param dataset: the dataset
        """
        self.roi = int(roi_id)
        self.t = int(t)
        self.target = int(target)
        self.dataset = int(dataset)
        project.repository.session.add(self)


def create_table() -> None:
    """
    Create the table to save results if it does not exist yet
    """
    with pydetecdiv_project(PyDetecDiv.project_name) as project:
        Base.metadata.create_all(project.repository.engine)


class TblClassNamesRow(tbl.IsDescription):
    """
    A class to describe the class names row saved in a table of an HDF5 file
    """
    class_name = tbl.StringCol(16)


def get_annotation_runs() -> dict:
    """
   Gets previously run prediction runs

    :return: a dictionary containing the ids of all prediction runs corresponding to a given list of classes
    """
    with pydetecdiv_project(PyDetecDiv.project_name) as project:
        results = list(project.repository.session.execute(
                sqlalchemy.text(f"SELECT run.id_,"
                                f"run.parameters ->> '$.annotator' as annotator, "
                                f"run.parameters ->> '$.class_names' as class_names "
                                f"FROM run "
                                f"WHERE (run.command='annotate_rois' OR run.command='import_annotated_rois') "
                                f"ORDER BY run.id_ ASC;")))
        runs = {}
        if results:
            for run in results:
                class_names = json.dumps(json.loads(run[2]))
                if class_names in runs:
                    runs[class_names].append(run[0])
                else:
                    runs[class_names] = [run[0]]
    return runs


def get_classifications(roi: ROI, run_list: list[int], as_index: bool = True) -> list[int] | list[str]:
    """
    Get the annotations for a ROI as defined in a list of runs

    :param roi: the ROI
    :param run_list: the list of Runs where the ROI was annotated or classified
    :param as_index: bool set to True to return annotations as indices of class_names list, set to False to return
     annotations as class names
    :return: the list of annotated classes by frame
    """
    roi_classes = [-1] * roi.fov.image_resource().image_resource_data().sizeT
    with pydetecdiv_project(PyDetecDiv.project_name) as project:
        results = list(project.repository.session.execute(
                sqlalchemy.text(f"SELECT rc.roi,rc.t,rc.class_name,"
                                f"run.parameters ->> '$.class_names' as class_names, rc.run, run.id_ "
                                f"FROM run, roi_classification as rc "
                                f"WHERE rc.run IN ({','.join([str(i) for i in run_list])}) and rc.roi={roi.id_} "
                                f"AND run.id_=rc.run "
                                f"ORDER BY rc.run ASC;")))
        if results:
            class_names = json.loads(results[0][3])
            if as_index:
                for annotation in results:
                    roi_classes[annotation[1]] = class_names.index(annotation[2])
            else:
                for annotation in results:
                    roi_classes[annotation[1]] = annotation[2]
    return roi_classes


class Plugin(plugins.Plugin):
    """
    A class extending plugins.Plugin to handle the example plugin
    """
    id_ = 'gmgm.plewniak.roiclasstorch'
    version = '1.0.0'
    name = 'ROI classification (PyTorch)'
    category = 'Deep learning'

    def __init__(self):
        super().__init__()
        self.menu = None

        self.parameters.parameter_list = [
            ChoiceParameter(name='model', label='Network', groups={'training', 'finetune', 'prediction'},
                            default='ResNet50V2_lstm', updater=self.load_models),
            ChoiceParameter(name='class_names', label='Classes',
                            groups={'training', 'finetune', 'prediction', 'annotate', 'import_annotations'},
                            updater=self.update_class_names),
            ChoiceParameter(name='weights', label='Weights', groups={'finetune', 'prediction'}, default='None',
                            updater=self.update_model_weights),
            IntParameter(name='seed', label='Random seed', groups={'training', 'finetune'}, maximum=999999999,
                         default=42),
            ChoiceParameter(name='optimizer', label='Optimizer', groups={'training', 'finetune'}, default='SGD',
                            items={'SGD'     : optim.SGD,
                                   'Adam'    : optim.Adam,
                                   'Adadelta': optim.Adadelta,
                                   'Adamax'  : optim.Adamax,
                                   'Nadam'   : optim.NAdam,
                                   }),
            FloatParameter(name='learning_rate', label='Learning rate', groups={'training', 'finetune'}, default=0.001,
                           minimum=0.00001, maximum=1.0),
            FloatParameter(name='decay_rate', label='Decay rate', groups={'training', 'finetune'}, default=0.95),
            IntParameter(name='decay_period', label='Decay period', groups={'training', 'finetune'}, default=2),
            FloatParameter(name='momentum', label='Momentum', groups={'training', 'finetune'}, default=0.9, ),
            ChoiceParameter(name='checkpoint_metric', label='Checkpoint metric', groups={'training', 'finetune'},
                            default='Loss', items={'Loss': 'val_loss', 'Accuracy': 'val_accuracy', }),
            CheckParameter(name='early_stopping', label='Early stopping', groups={'training', 'finetune'},
                           default=False),
            FloatParameter(name='num_training', label='Training dataset', groups={'training', 'finetune'}, default=0.6,
                           minimum=0.01, maximum=0.99, ),
            FloatParameter(name='num_validation', label='Validation dataset', groups={'training', 'finetune'},
                           default=0.2, minimum=0.01, maximum=0.99, ),
            FloatParameter(name='num_test', label='Test dataset', groups={'training', 'finetune'}, default=0.2,
                           minimum=0.01, maximum=0.99, decimals=2, ),
            IntParameter(name='dataset_seed', label='Random seed', groups={'training', 'finetune'}, default=42,
                         validator=lambda x: isinstance(x, int), maximum=999999999),
            ChoiceParameter(name='red_channel', label='Red', groups={'training', 'finetune', 'prediction'}, default=0,
                            updater=self.update_channels),
            ChoiceParameter(name='green_channel', label='Green', groups={'training', 'finetune', 'prediction'},
                            default=1, updater=self.update_channels),
            ChoiceParameter(name='blue_channel', label='Blue', groups={'training', 'finetune', 'prediction'}, default=2,
                            updater=self.update_channels),
            IntParameter(name='epochs', label='Epochs', groups={'training', 'finetune'}, default=16, ),
            IntParameter(name='batch_size', label='Batch size', groups={'training', 'finetune', 'prediction'},
                         default=128, ),
            IntParameter(name='seqlen', label='Sequence length', groups={'training', 'finetune', 'prediction'},
                         default=15, ),
            ItemParameter(name='annotation_file', label='Annotation file', groups={'import_annotations'}, ),
            ChoiceParameter(name='fov', label='Select FOVs', groups={'prediction'}, updater=self.update_fov_list),
            ]

        self.classifiers: ChoiceParameter = ChoiceParameter(name='classifier', label='Classifier',
                                                            groups={'import_classifier'})

    def register(self) -> None:
        """
        Registers the plugin
        """
        PyDetecDiv.app.project_selected.connect(self.update_parameters)
        PyDetecDiv.app.project_selected.connect(create_table)
        PyDetecDiv.app.viewer_roi_click.connect(self.add_context_action)

    def add_context_action(self, data: tuple[QGraphicsRectItem, QMenu]) -> None:
        """
        Adds an action to annotate the ROI from the FOV viewer

        :param data: the data sent by the PyDetecDiv().viewer_roi_click signal
        """
        r, menu = data
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            selected_roi = project.get_named_object('ROI', r.data(0))
            if selected_roi:
                roi_list = [selected_roi]
                annotate = menu.addAction('Annotate region classes')
                annotate.triggered.connect(lambda _: self.manual_annotation(roi_selection=roi_list))
                view_predictions = menu.addAction('View class predictions')
                view_predictions.triggered.connect(lambda _: self.show_results(roi_selection=roi_list))

    def addActions(self, menu: QMenu) -> None:
        """
        Overrides the addActions method in order to create a submenu with several actions for the same menu

        :param menu: the parent menu
        :type menu: QMenu
        """
        submenu = menu.addMenu(self.name)

        annotation_menu = submenu.addMenu('ROI Annotations')
        import_annoted_ROIs = QAction("Import annotations file", annotation_menu)
        manual_annotation = QAction("Annotate ROIs", annotation_menu)
        annotation_menu.addAction(import_annoted_ROIs)
        annotation_menu.addAction(manual_annotation)

        import_annoted_ROIs.triggered.connect(self.import_annotated_rois)
        manual_annotation.triggered.connect(self.manual_annotation)

        training_menu = submenu.addMenu('Train model')
        train_model = QAction("Train a model", training_menu)
        fine_tuning = QAction("Fine-tune training", training_menu)
        training_menu.addAction(train_model)
        training_menu.addAction(fine_tuning)

        train_model.triggered.connect(self.run_training)
        fine_tuning.triggered.connect(self.run_fine_tuning)

        predict_menu = submenu.addMenu('Classification')
        predict = QAction("Predict ROI classes", predict_menu)
        show_results = QAction("View classification results", predict_menu)
        export_classification = QAction("Export ROI classifications", predict_menu)
        predict_menu.addAction(predict)
        predict_menu.addAction(show_results)
        predict_menu.addAction(export_classification)

        predict.triggered.connect(self.run_prediction)
        show_results.triggered.connect(self.show_results)
        export_classification.triggered.connect(self.export_classification_to_csv)

        submenu.addSeparator()

        import_classifier = submenu.addAction('Import classifier')
        import_classifier.triggered.connect(self.run_import_classifier)

        submenu.aboutToShow.connect(
                lambda: self.set_enabled_actions(manual_annotation, train_model, fine_tuning, predict, show_results,
                                                 export_classification))

    def import_annotated_rois(self) -> None:
        """
        Selects a csv file containing ROI frames annotations and open a FOV2ROIlinks window to load the data it contains
        into the database as FOVs and ROIs with annotations.
        """
        filters = ["csv (*.csv)", "tsv (*.tsv)", ]
        annotation_file, _ = QFileDialog.getOpenFileName(PyDetecDiv.main_window,
                                                         caption='Choose file with annotated ROIs',
                                                         dir='.',
                                                         filter=";;".join(filters),
                                                         selectedFilter=filters[0])
        if annotation_file:
            self.parameters['annotation_file'].set_value(annotation_file)
            FOV2ROIlinks(annotation_file, self)

    def manual_annotation(self, trigger=None, roi_selection: list[ROI] = None, run: Run = None) -> None:
        """
        Opens a ManualAnnotator widget to annotate a list of ROIs

        :param trigger: the data passed by the triggered action
        :param roi_selection: the list of ROIs
        :param run: the current Run instance
        """
        annotation_runs = get_annotation_runs()
        annotator = ManualAnnotator()
        if roi_selection:
            annotator.set_roi_list(roi_selection)
        if annotation_runs:
            # self.parameters['class_names'].set_value(list(annotation_runs.keys())[0])
            self.parameters['class_names'].set_items({key: json.loads(key) for key in annotation_runs})
            self.parameters['class_names'].set_value(list(annotation_runs.keys())[0])
            annotator.setup(plugin=self)
            tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / ROI annotation')
            tab.project_name = PyDetecDiv.project_name
            tab.set_top_tab(annotator, 'Manual annotation')
            if roi_selection is None:
                annotator.update_ROI_selection(self.class_names())
            else:
                annotator.set_roi_list(roi_selection)
                annotator.next_roi()
            annotator.run = run
            annotator.setFocus()
        else:
            # annotator.setup(plugin=self)
            annotator.plugin = self
            suggestion = self.parameters['class_names'].value
            self.parameters['class_names'].clear()
            annotator.define_classes(suggestion=suggestion)

    def get_prediction_runs(self) -> dict:
        """
        Gets previously run prediction runs

        :return: a dictionary containing the ids of all prediction runs corresponding to a given list of classes
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = list(project.repository.session.execute(
                    sqlalchemy.text("SELECT run.id_,"
                                    "run.parameters ->> '$.class_names' as class_names "
                                    "FROM run "
                                    "WHERE run.command='predict' "
                                    "ORDER BY run.id_ ASC;")))
            runs = {}
            if results:
                for run in results:
                    class_names = json.dumps(json.loads(run[1]))
                    if class_names in runs:
                        runs[class_names].append(run[0])
                    else:
                        runs[class_names] = [run[0]]
                self.parameters['class_names'].set_items({class_names: json.loads(class_names) for class_names in runs})
        return runs

    def class_names(self, as_string: bool = True) -> list[str] | str:
        """
        Returns the classes currently in use

        :return: the class list
        """
        if as_string:
            return json.dumps(self.parameters['class_names'].value)
        return self.parameters['class_names'].value

    def update_class_names(self, prediction: bool = False) -> None:
        """
        Update the classes associated with the currently selected model
        """
        self.parameters['class_names'].clear()
        class_names = {}
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            run_list = project.get_objects('Run')
            if prediction:
                all_parameters = [run.parameters for run in run_list if run.command in ['predict']]
            else:
                all_parameters = [run.parameters for run in run_list if
                                  run.command in ['annotate_rois', 'import_annotated_rois', 'import_classifier']]

        for parameters in all_parameters:
            class_names.update({json.dumps(parameters['class_names']): parameters['class_names']})

        self.parameters['class_names'].set_items(class_names)

    def update_channels(self) -> None:
        """
        Updates the list of available channels to display in the GUI form
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            image_resource = project.get_object('ImageResource', 1)
            n_layers = image_resource.zdim if image_resource else 0

        for param in ['red_channel', 'green_channel', 'blue_channel']:
            self.parameters[param].set_items({str(i): i for i in range(n_layers)})

    def update_fov_list(self) -> None:
        """
        Updates the list of available FOV to display in the GUI form
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.parameters['fov'].set_items({fov.name: fov for fov in project.get_objects('FOV')})

    def update_model_weights(self, project_name: str = None) -> None:
        """
        Update the list of model weights associated with training and fine-tuning runs
        """
        if len(self.parameters['model'].items) == 0:
            self.load_models()
        self.parameters['weights'].clear()
        w_files = {}
        if project_name is None:
            project_name = PyDetecDiv.project_name
        with pydetecdiv_project(project_name) as project:
            run_list = project.get_objects('Run')
            all_parameters = [run.parameters for run in run_list if
                              run.command in ['train_model', 'fine_tune', 'import_classifier']]

        for parameters in all_parameters:
            module = self.parameters['model'].items[parameters['model']]
            run_weights = [parameters['best_weights'], parameters['last_weights']]
            model_path = module.__path__[0]
            w_files.update({f: os.path.join(model_path, f) for f in os.listdir(model_path) if
                            os.path.isfile(os.path.join(model_path, f)) and f in run_weights})
            try:
                user_path = os.path.join(get_project_dir(), 'roi_classification', 'models', parameters['model'])
                w_files.update({f: os.path.join(user_path, f) for f in os.listdir(user_path) if
                                os.path.isfile(os.path.join(user_path, f)) and f in run_weights})
            except FileNotFoundError:
                pass

        if w_files:
            self.parameters['weights'].set_items(w_files)

    def load_models(self) -> None:
        """
        Loads available models (modules)

        """
        available_models = {}
        for _, name, _ in pkgutil.iter_modules(models.__path__):
            available_models[name] = importlib.import_module(f'.models.{name}', package=__package__)
        for finder, name, _ in pkgutil.iter_modules([os.path.join(get_plugins_dir(), 'roi_classification/models')]):
            loader = finder.find_module(name)
            spec = importlib.util.spec_from_file_location(name, loader.path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            available_models[name] = module
        self.parameters['model'].set_items(available_models)

    def select_saved_parameters(self, weights_file: str) -> None:
        """
        Gets the parameters of a previous model training run, identified by the name of the saved weight file, for display in the
         GUI form

        :param weights_file: the weights file name
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            run_list = project.get_objects('Run')
        all_parameters = [run.parameters for run in run_list if
                          run.command in ['train_model', 'fine_tune', 'import_classifier']]
        for parameters in all_parameters:
            if weights_file in [parameters['best_weights'], parameters['last_weights']]:
                self.parameters['model'].value = parameters['model']
                self.parameters['class_names'].value = parameters['class_names']
                self.parameters['num_training'].value = parameters['num_training']
                self.parameters['num_validation'].value = parameters['num_validation']
                self.parameters['num_test'].value = 1.0 - parameters['num_training'] - parameters['num_validation']
                self.parameters['red_channel'].value = parameters['red_channel']
                self.parameters['green_channel'].value = parameters['green_channel']
                self.parameters['blue_channel'].value = parameters['blue_channel']
                self.parameters['dataset_seed'].value = parameters['dataset_seed']
                self.parameters['batch_size'].value = parameters['batch_size']
                self.parameters['seqlen'].value = parameters['seqlen']

    def set_enabled_actions(self, manual_annotation: QAction, train_model: QAction, fine_tuning: QAction, predict: QAction,
                            show_results: QAction, export_classification: QAction) -> None:
        """
        Enables or disables actions in menus according to the level of completion of the ROI classification workflow.

        :param manual_annotation: Manual annotation, enabled only if there are ROIs to annotate
        :param train_model: Train model, enabled only if there are annotated ROIs
        :param fine_tuning: Fine-tuning, enabled only if a previously trained model is available (previous training or import)
        :param predict: Prediction, enabled only if a previously trained model is available (previous training or import)
        :param show_results: Showing results, enabled only if a prediction has been run
        :param export_classification: Export classification, enabled only if a prediction has been run
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            manual_annotation.setEnabled(project.count_objects('ROI') > 0)
            train_model.setEnabled(len(self.get_annotated_rois(ids_only=True)) > 0)
            self.update_model_weights()
            fine_tuning.setEnabled(
                    (len(self.parameters['weights'].values) > 0) & (len(self.get_annotated_rois(ids_only=True)) > 0))
            predict.setEnabled(len(self.parameters['weights'].values) > 0)
            show_results.setEnabled(len(self.get_prediction_runs()) > 0)
            export_classification.setEnabled(
                    (len(self.get_annotated_rois(ids_only=True)) > 0) | (len(self.get_prediction_runs()) > 0))

    def get_annotated_rois(self, run: Run = None, ids_only: bool = False) -> list[ROI] | list[int]:
        """
        Gets a list of annotated ROI frames

        :return: the list of annotated ROI frames
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            db = QSqlDatabase("QSQLITE")
            db.setDatabaseName(project.repository.name)
            db.open()
            if run is None:
                if self.class_names(as_string=False):
                    query = QSqlQuery(
                            f"SELECT DISTINCT(roi) as annotated_rois FROM roi_classification, run "
                            f"WHERE run.id_=roi_classification.run "
                            f"AND (run.command='annotate_rois' OR run.command='import_annotated_rois') "
                            f"AND run.parameters ->> '$.class_names'=json('{self.class_names()}') ;",
                            db=db)
                else:
                    query = QSqlQuery(
                            "SELECT DISTINCT(roi) as annotated_rois FROM roi_classification, run "
                            "WHERE run.id_=roi_classification.run "
                            "AND (run.command='annotate_rois' OR run.command='import_annotated_rois') ",
                            db=db)
            else:
                if isinstance(run, int):
                    run = project.get_object('Run', run)
                query = QSqlQuery(
                        f"SELECT DISTINCT(roi) as annotated_rois FROM roi_classification, run "
                        f"WHERE run.id_=roi_classification.run "
                        f"AND run.id_={run.id_} ;",
                        db=db)
            query.exec()
            if query.first():
                roi_ids = [query.value('annotated_rois')]
                while query.next():
                    roi_ids.append(query.value('annotated_rois'))
                if ids_only is False:
                    return project.get_objects('ROI', roi_ids)
                return roi_ids
            return []

    def run_training(self) -> None:
        """
        Runs a model training process. Opens the TrainingDialog widget
        """
        if len(self.get_annotated_rois()) == 0:
            QMessageBox.critical(PyDetecDiv.main_window, 'No annotated ROI',
                                 'You should provide ground truth annotations for ROIs before training a model. '
                                 + 'Please, annotate ROIs or import annotations from a csv file.')
        else:
            TrainingDialog(self)

    def run_fine_tuning(self) -> None:
        """
        Runs fine-tuning process. Opens the FineTuningDialog widget
        """
        self.update_parameters(groups='finetune')
        self.update_model_weights()
        if len(self.parameters['weights'].values) == 0:
            QMessageBox.critical(PyDetecDiv.main_window, 'No classifier to refine',
                                 'You need a trained model for fine-tuning. '
                                 + 'Please, train a model first or import a classifier from another project.')
        else:
            FineTuningDialog(self)

    def run_prediction(self) -> None:
        """
        Runs a prediction process. Opens the PredictionDialog widget
        """
        PredictionDialog(self)

    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'running training on {"GPU" if device.type == "cuda" else "CPU"}')
        module = self.parameters['model'].value
        print(module.__name__)
        model = module.model.NN_module(len(self.parameters['class_names'].value)).to(device)

        print(f'{model.expected_shape}')

        if len(model.expected_shape) == 5:
            seqlen = self.parameters['seqlen'].value
            print(f'{datetime.now().strftime("%H:%M:%S")}: Sequence length: {seqlen}\n')
        else:
            seqlen = 0

        # os.makedirs(os.path.join(get_project_dir(), 'roi_classification', 'data'), exist_ok=True)
        # hdf5_file = os.path.join(get_project_dir(), 'roi_classification', 'data', 'annotated_rois.h5')
        # print(hdf5_file)
        #
        # z_channels = [self.parameters['red_channel'].value, self.parameters['green_channel'].value,
        #               self.parameters['blue_channel'].value]
        #
        # if not os.path.exists(hdf5_file):
        #     self.create_hdf5_annotated_rois(hdf5_file, z_channels=z_channels)
        #
        # print(f'{datetime.now().strftime("%H:%M:%S")}: Preparing data for training')
        #
        # training_idx, validation_idx, test_idx = prepare_data_for_training(hdf5_file, seqlen=seqlen,
        #                                                                    train=self.parameters[
        #                                                                        'num_training'].value,
        #                                                                    validation=self.parameters[
        #                                                                        'num_validation'].value,
        #                                                                    seed=self.parameters[
        #                                                                        'dataset_seed'].value)
        #
        # print(f'training: {len(training_idx)} validation: {len(validation_idx)} test: {len(test_idx)}')

        # train_loop(training_loader, validation_loader, model, loss_fn, optimizer, device, lambda1, lambda2)

    def show_results(self, trigger=None, roi_selection: list[ROI] = None) -> None:
        """
        Show predictions results in an Annotator widget

        :param trigger: the data passed by the triggered action
        :param roi_selection: the list of ROIs to show results for
        """
        prediction_runs = self.get_prediction_runs()
        if prediction_runs:
            self.parameters['class_names'].set_value(list(prediction_runs.keys())[0])
            tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / ROI class predictions')
            tab.project_name = PyDetecDiv.project_name
            annotator = PredictionViewer()
            annotator.setup(plugin=self)
            tab.set_top_tab(annotator, 'Prediction viewer')
            if roi_selection is None:
                annotator.update_ROI_selection(self.class_names())
            else:
                annotator.set_roi_list(roi_selection)
                annotator.next_roi()
            annotator.setFocus()
        else:
            QMessageBox.information(PyDetecDiv.main_window, 'Nothing to display',
                                    'There are no prediction results available for this project')

    def run_import_classifier(self) -> None:
        """
        Gets all runs with an available classifier from another project, and launches the ImportClassifierDialog for the
         user to choose one
        """
        current_project_name = PyDetecDiv.project_name
        self.classifiers.clear()
        for project_name in [p for p in project_list() if p != current_project_name]:
            with pydetecdiv_project(project_name) as project:
                run_list: list[Run] = [run for run in project.get_objects('Run') if run.command in ['train_model', 'fine_tune']]
                for run in run_list:
                    run.parameters['project'] = project_name
                    run.parameters['run'] = run.id_
                    run.command = 'import_classifier'
                    self.classifiers.add_item(
                            {f"{project_name}-{run.id_} {run.parameters['model']} {run.parameters['class_names']}": run})
        with pydetecdiv_project(current_project_name) as project:  # resetting global project name
            pass
        ImportClassifierDialog(self)

    def export_classification_to_csv(self, trigger, filename: str = 'classification.csv', roi_selection: list[int] = None,
                                     ground_truth: bool = True, run_list: list[int] = None) -> None:
        """
        Exports classification of ROIs in a CSV file

        :param trigger: the data passed by the triggered action
        :param filename: the CSV file name
        :param roi_selection: the list of ROI indices to include in the CSV file
        :param ground_truth: if True, ground truth classification is saved in the file
        :param run_list: the list of Run indices to export classification for
        """
        print('export ROI classification')
        df = self.get_classification_df(roi_selection=roi_selection, ground_truth=ground_truth, run_list=run_list)
        df.to_csv(os.path.join(get_project_dir(), filename))

    def get_classification_df(self, roi_selection: list[int] = None, ground_truth: bool = True,
                              run_list: list[int] = None) -> pd.DataFrame:
        """
        Gets classification in a pandas DataFrame

        :param roi_selection: the list of ROI indices to save
        :param ground_truth: if True, includes ground truth
        :param run_list: the list of Run ids to include
        :return: the pandas DataFrame with the annotations
        """
        annotation_runs = get_annotation_runs()
        if run_list is None:
            run_list = self.get_prediction_runs()[self.class_names()]
        annotations = {}
        if ground_truth:
            for roi in self.get_annotated_rois(ids_only=False):
                if roi_selection is None or (roi.id_ in roi_selection and annotation_runs):
                    annotations[roi.name] = get_classifications(roi=roi, run_list=annotation_runs[self.class_names()])
            df = pd.DataFrame(
                    [[roi_name, frame, label] for roi_name, v in annotations.items() for frame, label in enumerate(v)],
                    columns=('roi', 'frame', 'ground truth'))
        else:
            df = pd.DataFrame([], columns=('roi', 'frame'))

        for run in run_list:
            predictions = {}
            for roi in self.get_annotated_rois(run=run, ids_only=False):
                if roi_selection is None or (roi.id_ in roi_selection):
                    predictions[roi.name] = get_classifications(roi=roi, run_list=[run])

            predictions_df = pd.DataFrame(
                    [[roi_name, frame, label] for roi_name in predictions for frame, label in enumerate(predictions[roi_name])],
                    columns=('roi', 'frame', f'run_{run}'))
            if len(df) > 0:
                df = df.merge(predictions_df, on=['roi', 'frame'], how='outer')
            else:
                df = predictions_df
        return df

    def create_hdf5_annotated_rois(self, hdf5_file: str, z_channels: list[int] = None, channel: int = 0) -> None:
        """
        Creates a HDF5 file containing the annotated ROI data and their targets

        :param hdf5_file: the HDF5 file name
        :param z_channels: the z layers to be used as channels
        :param channel: the original channel to use
        """
        print(f'{datetime.now().strftime("%H:%M:%S")}: Retrieving data for annotated ROIs')
        data = self.get_annotations()
        print(f'{datetime.now().strftime("%H:%M:%S")}: Data retrieved with {len(data)} rows')

        print(f'{datetime.now().strftime("%H:%M:%S")}: Creating HDF5 file')
        h5file = tbl.open_file(hdf5_file, mode='w', title='ROI annotations')

        print(f'{datetime.now().strftime("%H:%M:%S")}: Getting fov data')
        fov_data = self.get_fov_data(z_layers=z_channels)
        mask = fov_data[['fov', 't']].apply(tuple, axis=1).isin(data[['fov', 't']].apply(tuple, axis=1))
        fov_data = fov_data[mask]

        print(f'{datetime.now().strftime("%H:%M:%S")}: Getting drift correction')
        drift_correction = self.get_drift_corrections()
        mask = drift_correction[['fov', 't']].apply(tuple, axis=1).isin(data[['fov', 't']].apply(tuple, axis=1))
        drift_correction = drift_correction[mask]

        print(f'{datetime.now().strftime("%H:%M:%S")}: Getting roi list')
        roi_list = self.get_roi_list()

        print(f'{datetime.now().strftime("%H:%M:%S")}: Applying drift correction to ROIs')
        roi_list = roi_list[roi_list['roi'].isin(set(data['roi']))]
        roi_list = pd.merge(drift_correction, roi_list, on=['fov'], how='left').dropna()
        roi_list['x0'] = (roi_list['x0'] + roi_list['dx'].round().astype(int))
        roi_list['x1'] = (roi_list['x1'] + roi_list['dx'].round().astype(int))
        roi_list['y0'] = (roi_list['y0'] + roi_list['dy'].round().astype(int))
        roi_list['y1'] = (roi_list['y1'] + roi_list['dy'].round().astype(int))

        width = (roi_list['x1'] - roi_list['x0'] + 1).max()
        height = (roi_list['y1'] - roi_list['y0'] + 1).max()

        print(f'{datetime.now().strftime("%H:%M:%S")}: FOV = {len(fov_data["fov"].unique())}', file=sys.stderr)
        print(f'{datetime.now().strftime("%H:%M:%S")}: T = {np.max(fov_data["t"]) + 1}', file=sys.stderr)
        print(f'{datetime.now().strftime("%H:%M:%S")}: ROIs = {len(roi_list["roi"].unique())} ({len(roi_list)})',
              file=sys.stderr)

        roi_values = np.array(roi_list["roi"])
        roi_list["roi"], roi_mapping = fastremap.renumber(roi_values, in_place=False, preserve_zero=False)
        num_rois = len(roi_mapping)
        num_frames = np.max(fov_data['t']) + 1

        print(f'{datetime.now().strftime("%H:%M:%S")}: Creating target datasets')

        targets = data.loc[:, ['t', 'roi', 'class_name']]
        targets['roi'] = fastremap.remap(np.array(targets['roi']), roi_mapping)
        targets['label'] = targets['class_name'].apply(lambda x: self.class_names(as_string=False).index(x))

        initial_values = np.zeros((num_frames, num_rois,), dtype=np.int8) - 1
        target_array = h5file.create_carray(h5file.root, 'targets', atom=tbl.Int8Atom(), shape=(num_frames, num_rois),
                                            chunkshape=(num_frames, num_rois,), obj=initial_values)

        class_names_table = h5file.create_table(h5file.root, 'class_names', TblClassNamesRow, 'Class names')
        class_names_table.append([(name,) for name in self.class_names(as_string=False)])

        print(f'{datetime.now().strftime("%H:%M:%S")}: Creating ROI dataset')

        roi_data = h5file.create_carray(h5file.root, 'roi_data', atom=tbl.Float16Atom(shape=(height, width, 3)),
                                        chunkshape=(50, num_rois,), shape=(num_frames, num_rois))

        print(f'{datetime.now().strftime("%H:%M:%S")}: Reading and compositing images')
        for row in fov_data.itertuples():
            if row.t % 10 == 0:
                print(f'{datetime.now().strftime("%H:%M:%S")}: FOV {row.fov}, frame {row.t}')

            rois = roi_list.loc[(roi_list['fov'] == row.fov) & (roi_list['t'] == row.t)]

            # If merging and normalization are too slow, maybe use tensorflow or pytorch to do the operations
            fov_img = cv2.merge([cv2.imread(z_file, cv2.IMREAD_UNCHANGED) for z_file in reversed(row.channel_files)])

            for roi in rois.itertuples():
                roi_data[row.t, roi.roi - 1, ...] = cv2.normalize(fov_img[roi.y0:roi.y1 + 1, roi.x0:roi.x1 + 1],
                                                                  dtype=cv2.CV_16F, dst=None, alpha=1e-10, beta=1.0,
                                                                  norm_type=cv2.NORM_MINMAX)
                target_array[row.t, roi.roi - 1] = targets.loc[
                    (targets['t'] == row.t) & (targets['roi'] == roi.roi), 'label'].values

        h5file.close()
        print(f'{datetime.now().strftime("%H:%M:%S")}: HDF5 file of annotated ROIs ready')
