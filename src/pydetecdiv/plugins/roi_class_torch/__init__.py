import gc
import importlib
import json
import os
import pkgutil
import random
import sys
from datetime import datetime

import cv2
import fastremap
import polars
import tables as tbl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

import sqlalchemy
import torch
from PySide6.QtGui import QAction
from PySide6.QtSql import QSqlDatabase, QSqlQuery
from PySide6.QtWidgets import QMenu, QFileDialog, QMessageBox, QGraphicsRectItem

from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import registry
from sqlalchemy.types import JSON
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torchinfo import summary

from torch import optim

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, get_project_dir, project_list
from pydetecdiv.domain.Run import Run
from pydetecdiv.domain.Project import Project
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.plugins.parameters import ItemParameter, ChoiceParameter, IntParameter, FloatParameter, CheckParameter

from . import models
from .data import prepare_data_for_training, ROIDataset
from .evaluate import evaluate_metrics, evaluate_model
from .gui.ImportAnnotatedROIs import FOV2ROIlinks
from .gui.classification import ManualAnnotator, PredictionViewer, DefineClassesDialog
from .gui.modelinfo import ModelInfoDialog
from .gui.prediction import PredictionDialog
from .gui.training import TrainingDialog, FineTuningDialog, ImportClassifierDialog

from pydetecdiv.settings import get_plugins_dir, get_config_value
from .training import train_testing_loop, train_loop
from ...app.gui.core.widgets.viewers.plots import MatplotViewer
from ...domain.Dataset import Dataset

Base = registry().generate_base()


class Results(Base):
    """
    The DAO defining and handling the table to store results
    """
    __tablename__ = 'roi_classification'
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


class TblRoiNamesRow(tbl.IsDescription):
    """
    A class to describe the ROI names row saved in a table of an HDF5 file
    """
    roi_name = tbl.StringCol(32)


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


def get_roi_list() -> pd.DataFrame:
    """
    Gets a list of ROIs in a DataFrame

    :return: pandas DataFrame containing ROIO id, FOV id, ROI x, y positions of top left and bottom right corners
    """
    with pydetecdiv_project(PyDetecDiv.project_name) as project:
        results = pd.DataFrame(project.repository.session.execute(
                sqlalchemy.text("SELECT id_ as roi, name, fov, x0_ as x0, y0_ as y0, x1_ as x1, y1_ as y1 "
                                "FROM ROI "
                                "ORDER BY fov, id_ ASC;")))
    return results


def get_drift_corrections() -> pd.DataFrame:
    """
    Gets the drift correction values for FOVs in a pandas DataFrame

    :return: a pandas DataFrame with the (dx, dy) drift values for each FOV and frame
    """
    with pydetecdiv_project(PyDetecDiv.project_name) as project:
        results = pd.DataFrame(project.repository.session.execute(
                sqlalchemy.text("SELECT fov, img.key_val ->> '$.drift' as drift "
                                "FROM ImageResource as img "
                                "ORDER BY fov ASC;")))
        drift_corrections = pd.DataFrame(columns=['fov', 't', 'dx', 'dy'])
        for row in results.itertuples(index=False):
            if row.drift is not None:
                df = pd.read_csv(os.path.join(get_project_dir(), row.drift))
                df['fov'] = row.fov
                df['t'] = df.index
                drift_corrections = pd.concat([drift_corrections, df], ignore_index=True)
        return drift_corrections


def save_training_datasets(run, hdf5_file: str, training_idx: list[int], validation_idx: list[int],
                           test_idx: list[int]) -> None:
    """
    Saves in TrainingData table the (ROI, frame, class) data subsets that were used for training, validation and testing while
    training a model.

    :param hdf5_file: the HDF5 file containing the annotated ROI data
    :param training_idx: the list of (ROI, frame) data indices in the training dataset
    :param validation_idx: the list of (ROI, frame) data indices in the validation dataset
    :param test_idx: the list of (ROI, frame) data indices in the validation dataset
    """
    training_ds = Dataset(project=run.project, name=f'train_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                          type_='training', run=run.id_)
    validation_ds = Dataset(project=run.project, name=f'val_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                            type_='validation', run=run.id_)
    test_ds = Dataset(project=run.project, name=f'test_{datetime.now().strftime("%Y%m%d-%H%M%S")}', type_='test',
                      run=run.id_)

    # print(len(training_idx), len(validation_idx), len(test_idx))

    h5file = tbl.open_file(hdf5_file, mode='r')
    targets = h5file.root.targets.read()
    h5file.close()

    for frame, roi_id in training_idx:
        TrainingData().save(run.project, roi_id, frame, targets[frame, roi_id], training_ds.id_)

    for frame, roi_id in validation_idx:
        TrainingData().save(run.project, roi_id, frame, targets[frame, roi_id], validation_ds.id_)

    for frame, roi_id in test_idx:
        TrainingData().save(run.project, roi_id, frame, targets[frame, roi_id], test_ds.id_)

    run.project.commit()


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
            ChoiceParameter(name='model', label='Network', groups={'training', 'finetune', 'prediction', 'info'},
                            default='ResNet18_lstm', updater=self.load_models),
            ChoiceParameter(name='class_names', label='Classes',
                            groups={'training', 'finetune', 'prediction', 'annotate', 'import_annotations', 'info'},
                            updater=self.update_class_names),
            ChoiceParameter(name='weights', label='Weights', groups={'finetune', 'prediction'}, default='None',
                            updater=self.update_model_weights),
            IntParameter(name='seed', label='Random seed', groups={'training', 'finetune'}, maximum=999999999,
                         default=42),
            ChoiceParameter(name='optimizer', label='Optimizer', groups={'training', 'finetune'}, default='AdamW',
                            items={'AdamW'   : optim.AdamW,
                                   'SGD'     : optim.SGD,
                                   'Adam'    : optim.Adam,
                                   'Adadelta': optim.Adadelta,
                                   'Adamax'  : optim.Adamax,
                                   'Nadam'   : optim.NAdam,
                                   }),
            FloatParameter(name='learning_rate', label='Learning rate', groups={'training', 'finetune'}, default=1e-4,
                           minimum=1e-6, maximum=1.0),
            FloatParameter(name='decay_rate', label='Decay rate', groups={'training', 'finetune'}, default=0.95),
            IntParameter(name='decay_period', label='Decay period', groups={'training', 'finetune'}, default=50),
            FloatParameter(name='weight_decay', label='Weight decay', groups={'training', 'finetune'}, default=1e-2, ),
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

        submenu.addSeparator()
        show_model_info = submenu.addAction('Show model info')
        show_model_info.triggered.connect(self.run_model_info)

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

    def save_annotations(self, roi: ROI, roi_classes: list[int], run) -> None:
        """
        Saves manual annotation into the database

        :param roi: the annotated ROI
        :param roi_classes: the classes along time
        :param run: the annotation run
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            for t, class_id in enumerate(roi_classes):
                if class_id != -1:
                    Results().save(project, run, roi, t, pd.DataFrame([1]), [self.class_names(as_string=False)[class_id]])

    def save_annotation(self, project: Project, run: Run, roi: ROI, frame: int, class_name: str) -> None:
        """
        Saves the results in database

        :param project: the current project
        :param run: the current run
        :param roi: the current ROI
        :param frame: the current frame
        :param class_name: the class name
        """
        if roi.sizeT > frame:
            Results().save(project, run, roi, frame, np.array([int(class_name == c) for c in self.class_names(as_string=False)]),
                           self.class_names(as_string=False))

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

    def get_unannotated_rois(self) -> (list[ROI], list[ROI]):
        """
        Gets the unannotated ROIs

        :return: list of unannotated ROIs and list of all ROIs
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            all_roi_ids = [roi.id_ for roi in project.get_objects('ROI')]
            print(f'All ROIs: {len(all_roi_ids)}')
            annotated_rois = self.get_annotated_rois(ids_only=True)
            print(f'Annotated ROIs: {len(annotated_rois)}')
            unannotated_roi_ids = set(all_roi_ids).difference(set(annotated_rois))
            print(f'Unannotated ROIs: {len(unannotated_roi_ids)}')
            return project.get_objects('ROI', list(unannotated_roi_ids)), project.get_objects('ROI', list(all_roi_ids))

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

    def run_model_info(self) -> None:
        """
        Shows model information. Opens the ModelInfoDialog widget
        """
        ModelInfoDialog(self)

    def load_model(self, pretrained=False):
        print(f'{datetime.now().strftime("%H:%M:%S")}: Model: {self.parameters["model"].key}\n')
        if pretrained:
            model = torch.jit.load(os.path.join(get_project_dir(), 'roi_classification', 'models', self.parameters['model'].key,
                                                self.parameters['weights'].value))
            return model, self.parameters['model'].key
        return (self.parameters['model'].value.model.NN_module(len(self.parameters['class_names'].value)),
                self.parameters['model'].key)

    def get_input_shape(self, model):
        seqlen = 0
        if len(model.expected_shape) == 5:
            seqlen = self.parameters['seqlen'].value
            print(f'{datetime.now().strftime("%H:%M:%S")}: Sequence length: {seqlen}\n')
            img_size = model.expected_shape[3:5]
        else:
            img_size = model.expected_shape[2:4]
        print(f'{datetime.now().strftime("%H:%M:%S")}: Input image size: {img_size}\n')
        return img_size, seqlen

    def get_weights_filepaths(self, run):
        os.makedirs(os.path.join(get_project_dir(), 'roi_classification', 'models', self.parameters['model'].key), exist_ok=True)
        checkpoint_monitor_metric = self.parameters['checkpoint_metric'].value
        best_checkpoint_filename = f'{run.id_}_best_{checkpoint_monitor_metric}.weights.pt'
        checkpoint_filepath = os.path.join(get_project_dir(), 'roi_classification', 'models',
                                           self.parameters['model'].key,
                                           f'{best_checkpoint_filename}')
        last_weights_filename = f'{run.id_}_last.weights.pt'
        last_weights_filepath = os.path.join(get_project_dir(), 'roi_classification', 'models', self.parameters['model'].key,
                                             last_weights_filename)
        return checkpoint_filepath, last_weights_filepath

    def save_training_run(self, fine_tuning: bool = False) -> Run:
        """
        Saves the current training Run

        :param finetune: False if the run is a training run (from scratch or pretrained Keras model), True if it is a fine-tuning
         run
        :return: the current Run instance
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            if fine_tuning:
                return self.save_run(project, 'fine_tune', self.parameters.json(groups='finetune'))
            return self.save_run(project, 'train_model', self.parameters.json(groups='training'))

    def train_model(self, fine_tuning=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.random.manual_seed(self.parameters['seed'].value)

        print(f'running training on {"GPU" if device.type == "cuda" else "CPU"}')

        model, model_name = self.load_model(pretrained=fine_tuning)
        img_size, seqlen = self.get_input_shape(model)

        hdf5_file = self.create_hdf5_annotated_rois()

        print(f'{datetime.now().strftime("%H:%M:%S")}: Preparing data for training')

        training_idx, validation_idx, test_idx, class_weights = prepare_data_for_training(hdf5_file, seqlen=seqlen,
                                                                           train=self.parameters[
                                                                               'num_training'].value,
                                                                           validation=self.parameters[
                                                                               'num_validation'].value,
                                                                           seed=self.parameters[
                                                                               'dataset_seed'].value)

        print(f'training: {len(training_idx)} validation: {len(validation_idx)} test: {len(test_idx)}')

        n_epochs = self.parameters['epochs'].value
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='mean')
        # loss_fn = torch.nn.BCELoss()
        lr = self.parameters['learning_rate'].value
        weight_decay = self.parameters['weight_decay'].value
        decay_rate = self.parameters['decay_rate'].value
        decay_period = self.parameters['decay_period'].value
        momentum = self.parameters['momentum'].value
        lambda1, lambda2 = 0.0, 0.0
        seq2one = False

        ### Make sure weight decay is only applied to Linear and Conv2d layers, as it should not be applied to Batch normalization
        ### and possibly other normalization layers
        ### but does not work... get error: TypeError: optimizer can only optimize Tensors, but one of the params is NoneType
        # no_decay = list()
        # decay = list()
        # for m in model.modules():
        #   if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d,)):
        #     decay.append(m.weight)
        #     no_decay.append(m.bias)
        #   elif hasattr(m, 'weight'):
        #     no_decay.append(m.weight)
        #   elif hasattr(m, 'bias'):
        #     no_decay.append(m.bias)
        # for name, param in model.named_modules():
        #   if not (name.endswith('.weight') or name.endswith('.bias')):
        #     no_decay.append(param)
        # model_param = [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': weight_decay}]

        model_param = model.parameters()

        match self.parameters['optimizer'].key:
            case 'Adam':
                optimizer = self.parameters['optimizer'].value(model_param, lr=lr, weight_decay=weight_decay)
            case 'AdamW':
                optimizer = self.parameters['optimizer'].value(model_param, lr=lr, weight_decay=weight_decay)
            case 'SGD':
                optimizer = self.parameters['optimizer'].value(model_param, lr=lr, momentum=momentum,
                                                               weight_decay=weight_decay)

        training_dataset = ROIDataset(hdf5_file, training_idx, targets=True, image_shape=img_size, seq2one=seq2one, seqlen=seqlen)
        validation_dataset = ROIDataset(hdf5_file, validation_idx, targets=True, image_shape=img_size, seq2one=seq2one, seqlen=seqlen)
        test_dataset = ROIDataset(hdf5_file, test_idx, targets=True, image_shape=img_size, seq2one=seq2one, seqlen=seqlen)

        run = self.save_training_run(fine_tuning=fine_tuning)

        save_training_datasets(run, hdf5_file, training_idx, validation_idx, test_idx)

        batch_size = self.parameters['batch_size'].value

        train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        step_scheduler = StepLR(optimizer, step_size=decay_period, gamma=decay_rate, last_epoch=-1)
        # scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        history = polars.DataFrame(schema={'train loss'    : polars.datatypes.Float64, 'val loss': polars.datatypes.Float64,
                                           'train accuracy': polars.datatypes.Float64, 'val accuracy': polars.datatypes.Float64,
                                           })
        model = model.to(device)

        checkpoint_filepath, last_weights_filepath = self.get_weights_filepaths(run)

        if fine_tuning:
            min_val_loss, accuracy = evaluate_metrics(model, validation_dataloader, seq2one, loss_fn, lambda1, lambda2, device)
            print(f'Fine tuning starting with validation loss = {min_val_loss} and initial accuracy = {accuracy}')
        else:
            min_val_loss = torch.finfo(torch.float).max

        for epoch in range(n_epochs):
            history.extend(train_loop(train_dataloader, validation_dataloader, model, seq2one,
                                      loss_fn, optimizer, lambda1, lambda2, device))
            if history['val loss'][-1] < min_val_loss:
                min_val_loss = history['val loss'][-1]
                model_scripted = torch.jit.script(model)
                model_scripted.save(checkpoint_filepath)
                print(f"Saving best model at epoch {epoch + 1} with val loss {min_val_loss}"
                      f" and train loss {history['train loss'][-1]}")
                run.parameters.update({'best_weights': os.path.basename(checkpoint_filepath), 'best_epoch': epoch + 1})
                run.validate().commit()

            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"Training Loss: {history['train loss'][-1]:.4f}, "
                  f"Validation Loss: {history['val loss'][-1]:.4f}, "
                  f"Accuracy: {100 * history['train accuracy'][-1]:.1f} %, "
                  f"Val accuracy: {100 * history['val accuracy'][-1]:.1f} %, "
                  f"learning rate: {scheduler.get_last_lr()[0]}, "
                  f" -- ({datetime.now().strftime('%H:%M:%S')})")
            # f"learning rate: {scheduler.get_last_lr()}, ")
            step_scheduler.step()
            scheduler.step(history['val loss'][-1])

        ##################################################################
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(last_weights_filepath)  # Save
        del (model_scripted)
        gc.collect()

        run.parameters.update({'last_weights': os.path.basename(last_weights_filepath)})
        run.validate().commit()

        print(f'{datetime.now().strftime("%H:%M:%S")}: Evaluation on test dataset')
        avg_test_loss, test_accuracy = evaluate_metrics(model, test_dataloader, seq2one, loss_fn, lambda1, lambda2, device)
        evaluation = {'loss': avg_test_loss, 'accuracy': test_accuracy}
        print(f"Test loss: {avg_test_loss:.4f}, "
              f"Test accuracy: {100 * test_accuracy:.1f} %, ")

        stats, ground_truth, predictions, best_ground_truth, best_predictions = evaluate_model(model, checkpoint_filepath,
                                                                                               self.parameters['class_names'].value,
                                                                                               test_dataloader,
                                                                                               seqlen, seq2one, device)
        # del (model)
        # torch.cuda.empty_cache()
        # gc.collect()

        if run.key_val is None:
            run.key_val = stats
        else:
            run.key_val.update(stats)

        run.validate().commit()

        print(f'{datetime.now().strftime("%H:%M:%S")}: Statistics for last model:', file=sys.stderr)
        print(polars.DataFrame(stats['last_stats']), file=sys.stderr)

        print(f'{datetime.now().strftime("%H:%M:%S")}: Statistics for best model:', file=sys.stderr)
        print(polars.DataFrame(stats['best_stats']), file=sys.stderr)

        datasets = {'train': training_dataset, 'val': validation_dataset, 'test': test_dataset}

        return (model_name, self.parameters['class_names'].value, history, evaluation,
                ground_truth, predictions, best_ground_truth, best_predictions, datasets, model, device)

    def predict(self) -> None:
        """
        Running prediction on all ROIs in selected FOVs.
        """
        seqlen = self.parameters['seqlen'].value
        z_channels = (self.parameters['red_channel'].value, self.parameters['green_channel'].value,
                      self.parameters['blue_channel'].value,)
        fov_names = [self.parameters['fov'].key]
        module = self.parameters['model'].value
        print(module.__name__)

        seq2one = False

        print(f'{datetime.now().strftime("%H:%M:%S")}: Loading weights')
        model = torch.jit.load(self.parameters['weights'].value)
        img_size = (model.expected_shape[-2], model.expected_shape[-1])

        # print(f'{datetime.now().strftime("%H:%M:%S")}: Preparing data')
        # hdf5_file = self.create_hdf5_unannotated_rois(overwrite=False)


        # pred_dataset = ROIDataset(hdf5_file, training_idx, targets=True, image_shape=img_size, seq2one=seq2one, seqlen=seqlen)



    def prepare_data_for_classification(self, fov_list: list[int],
                                        z_channels: tuple[int] = None) -> (pd.DataFrame, pd.DataFrame, np.ndarray):
        """
        Prepares the data for class prediction. Drift correction is automatically applied

        :param fov_list: the list of FOV indices whose ROIs should be classified
        :param z_channels: the z layers to be used as channels
        :return: Pandas DataFrames containing FOV data, list of (ROI, frame) with positions, unique indices of ROIs
        """
        print(f'{datetime.now().strftime("%H:%M:%S")}: Getting fov data')
        fov_data = self.get_fov_data(z_layers=z_channels)
        mask = fov_data['fov'].isin(fov_list)
        fov_data = fov_data[mask]

        print(f'{datetime.now().strftime("%H:%M:%S")}: Getting drift correction')
        drift_correction = self.get_drift_corrections()
        mask = drift_correction['fov'].isin(fov_list)
        drift_correction = drift_correction[mask]

        print(f'{datetime.now().strftime("%H:%M:%S")}: Getting roi list')
        roi_list = self.get_roi_list()

        print(f'{datetime.now().strftime("%H:%M:%S")}: Applying drift correction to ROIs')
        roi_list = pd.merge(drift_correction, roi_list, on=['fov'], how='left').dropna()
        roi_list['x0'] = (roi_list['x0'] + roi_list['dx'].round().astype(int))
        roi_list['x1'] = (roi_list['x1'] + roi_list['dx'].round().astype(int))
        roi_list['y0'] = (roi_list['y0'] + roi_list['dy'].round().astype(int))
        roi_list['y1'] = (roi_list['y1'] + roi_list['dy'].round().astype(int))

        rois = roi_list["roi"].unique()

        print(f'{datetime.now().strftime("%H:%M:%S")}: FOV = {len(fov_data["fov"].unique())}')
        print(f'{datetime.now().strftime("%H:%M:%S")}: T = {np.max(fov_data["t"]) + 1}')
        print(f'{datetime.now().strftime("%H:%M:%S")}: ROIs = {len(rois)} ({len(roi_list)})')

        return fov_data, roi_list, rois


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

    def create_hdf5_annotated_rois(self) -> str:
        """
        Creates a HDF5 file containing the annotated ROI data and their targets
        """
        os.makedirs(os.path.join(get_project_dir(), 'roi_classification', 'data'), exist_ok=True)
        hdf5_file = os.path.join(get_project_dir(), 'roi_classification', 'data', 'annotated_rois.h5')

        z_channels = (self.parameters['red_channel'].value, self.parameters['green_channel'].value,
                      self.parameters['blue_channel'].value)

        if not os.path.exists(hdf5_file):
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
            drift_correction = get_drift_corrections()
            mask = drift_correction[['fov', 't']].apply(tuple, axis=1).isin(data[['fov', 't']].apply(tuple, axis=1))
            drift_correction = drift_correction[mask]

            print(f'{datetime.now().strftime("%H:%M:%S")}: Getting roi list')
            roi_list = get_roi_list()

            roi_list = roi_list[roi_list['roi'].isin(set(data['roi']))]

            if not drift_correction.empty:
                print(f'{datetime.now().strftime("%H:%M:%S")}: Applying drift correction to ROIs')
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

            # print([(roi_name,) for roi_name in roi_list['name'].unique()], file=sys.stderr)
            roi_names = roi_list[['roi', 'name']].drop_duplicates()
            roi_names_table = h5file.create_table(h5file.root, 'roi_names', TblRoiNamesRow, 'ROI names')
            roi_names_table.append([(roi_name,) for roi_name in roi_list['name'].unique()])

            # h5file.create_carray(h5file.root, 'roi_names', atom=tbl.StringAtom(64), chunkshape=(50, num_rois,),
            #                                  shape=(num_frames, num_rois))

            print(f'{datetime.now().strftime("%H:%M:%S")}: Reading and compositing images')

            for row in fov_data.itertuples():
                if row.t % 10 == 0:
                    print(f'{datetime.now().strftime("%H:%M:%S")}: FOV {row.fov}, frame {row.t}')

                if 't' in roi_list:
                    rois = roi_list.loc[(roi_list['fov'] == row.fov) & (roi_list['t'] == row.t)]
                else:
                    rois = roi_list.loc[(roi_list['fov'] == row.fov)]

                # If merging and normalization are too slow, maybe use tensorflow or pytorch to do the operations
                fov_img = cv2.merge([cv2.imread(z_file, cv2.IMREAD_UNCHANGED) for z_file in reversed(row.channel_files)])

                for roi in rois.itertuples():
                    roi_data[row.t, roi.roi - 1, ...] = cv2.normalize(fov_img[roi.y0:roi.y1 + 1, roi.x0:roi.x1 + 1],
                                                                      dtype=cv2.CV_16F, dst=None, alpha=1e-10, beta=1.0,
                                                                      norm_type=cv2.NORM_MINMAX)
                    roi_names_table[roi.roi - 1] = roi_names.loc[roi_names['roi'] == roi.roi, 'name'].values
                    target_array[row.t, roi.roi - 1] = targets.loc[
                        (targets['t'] == row.t) & (targets['roi'] == roi.roi), 'label'].values

            h5file.close()
            print(f'{datetime.now().strftime("%H:%M:%S")}: HDF5 file  {hdf5_file} of annotated ROIs ready')
        else:
            print(f'{datetime.now().strftime("%H:%M:%S")}: Using existing HDF5 file {hdf5_file}')
        return hdf5_file

    def get_annotations(self) -> pd.DataFrame:
        """
        Gets ROI annotations from manual annotation or imported annotation runs in a DataFrame

        :return: a pandas DataFrame containing run id, ROI and FOV ids, frame, class name
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = pd.DataFrame(project.repository.session.execute(
                    sqlalchemy.text(f"SELECT run.id_, rc.roi, roi.fov, rc.t, rc.class_name "
                                    f"FROM roi_classification as rc, run, ROI as roi "
                                    f"WHERE (run.command='annotate_rois' OR run.command='import_annotated_rois') "
                                    f"AND rc.run=run.id_ AND roi.id_=rc.roi "
                                    f"AND run.parameters ->> '$.annotator'='{get_config_value('project', 'user')}' "
                                    f"AND run.parameters ->> '$.class_names'=json('{self.class_names()}') "
                                    f"ORDER BY run.id_, rc.t, rc.roi ASC;")))
            results = results.drop_duplicates(subset=['roi', 't'], keep='last')
            return results

    def get_fov_data(self, z_layers: tuple[int, int, int] | tuple[int] = None, channel: int = None) -> pd.DataFrame:
        """
        Gets FOV data, i.e. FOV id, time frame and the list of files with the z layers to be used as RGB channels

        :param z_layers: the z layer files to use as RGB channels
        :param channel: the channel of the original image to be loaded
        :return: a pandas DataFrame with the FOV data
        """
        if z_layers is None:
            z_layers = (0,)
        if channel is None:
            channel = 0
            # channel = self.parameters['channel']

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = pd.DataFrame(project.repository.session.execute(
                    sqlalchemy.text(f"SELECT img.fov, data.t, data.url, data.id_ "
                                    f"FROM data, ImageResource as img "
                                    f"WHERE data.image_resource=img.id_ "
                                    f"AND data.z in {tuple(z_layers)} "
                                    f"AND data.c={channel} "
                                    f"ORDER BY img.fov, data.url ASC;")))
            for row in results.itertuples():
                results.at[row.Index, 'url'] = project.get_object('Data', row.id_).url
            print(results, file=sys.stderr)
            fov_data = results.groupby(['fov', 't'])['url'].apply(self.layers2channels).reset_index()
        fov_data.columns = ['fov', 't', 'channel_files']
        return fov_data

    def layers2channels(self, zfiles) -> list[str]:
        """
        Gets the image file names for red, green and blue channels

        :param zfiles: the list of files corresponding to one z layer each
        :return: the list of zfiles
        """
        zfiles = list(zfiles)
        return [zfiles[i] for i in [self.parameters['red_channel'].value, self.parameters['green_channel'].value,
                                    self.parameters['blue_channel'].value]]
