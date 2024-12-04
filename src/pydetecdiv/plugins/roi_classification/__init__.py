"""
An plugin for deep-learning ROI classification
"""
import importlib
import json
import math
import os.path
import pkgutil
import sys
from collections import Counter
from datetime import datetime

import cv2
import fastremap
import keras.optimizers
import pandas as pd

import tables as tbl
import numpy as np
import sqlalchemy
from PySide6.QtGui import QAction, QColor
from PySide6.QtSql import QSqlDatabase, QSqlQuery
from PySide6.QtWidgets import QGraphicsRectItem, QFileDialog, QMessageBox
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import registry
from sqlalchemy.types import JSON
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

from pydetecdiv import plugins, copy_files
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, get_project_dir, project_list
from pydetecdiv.settings import get_plugins_dir
from pydetecdiv.domain import Dataset, Run, Project, ROI
from pydetecdiv.settings import get_config_value

from . import models
from .gui.ImportAnnotatedROIs import FOV2ROIlinks
from .gui.classification import ManualAnnotator, PredictionViewer, DefineClassesDialog
from .gui.prediction import PredictionDialog
from .gui.training import TrainingDialog, FineTuningDialog, ImportClassifierDialog
from ..parameters import ItemParameter, ChoiceParameter, IntParameter, FloatParameter, CheckParameter

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

    def save(self, project, run, roi, t, predictions, class_names):
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

    def save(self, project, roi_id, t, target, dataset):
        """
        Save the results from a plugin run on a ROI at time t into the database

        :param project: the current project
        :param run: the current run
        :param roi: the current ROI
        :param t: the current frame
        :param predictions: the list of prediction values
        :param class_names: the class names
        """
        self.roi = int(roi_id)
        self.t = int(t)
        self.target = int(target)
        self.dataset = int(dataset)
        project.repository.session.add(self)


class DataProvider(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, h5file_name, indices, batch_size=32, image_shape=(60, 60), seqlen=0, name=None, targets=False,
                 shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.h5file = tbl.open_file(h5file_name, mode='r')
        self.roi_data = self.h5file.root.roi_data
        if targets:
            self.targets = self.h5file.root.targets
        else:
            self.targets = None
        self.indices = indices
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.name = name
        self.image_shape = image_shape
        self.shuffle = shuffle
        self.on_epoch_end()  # Shuffle indices at the beginning of each epoch

    def __len__(self):
        """

        :return:
        """
        # Number of batches per epoch
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        if self.seqlen == 0:
            batch_roi_data = np.array([tf.image.resize(np.array(self.roi_data[frame, roi_id, ...]), self.image_shape,
                                                       method='nearest') for (frame, roi_id) in batch_indices])
            if self.targets is not None:
                batch_targets = np.array([self.targets[frame, roi_id] for (frame, roi_id) in batch_indices])
        else:
            batch_roi_data = np.array(
                    [[tf.image.resize(np.array(self.roi_data[frame + i, roi_id, ...]), self.image_shape,
                                      method='nearest') for i in range(self.seqlen)] for frame, roi_id in batch_indices])
            if self.targets is not None:
                batch_targets = np.array([self.targets[frame:frame + self.seqlen, roi_id] for (frame, roi_id) in batch_indices])

        # print(f'{self.name} {index}: {batch_roi_data.shape} {batch_targets.shape}')
        if self.targets is not None:
            return batch_roi_data, batch_targets
        return batch_roi_data

    def on_epoch_end(self):
        """

        """
        if self.shuffle:
            np.random.shuffle(self.indices)

    def close(self):
        """

        """
        self.h5file.close()


class PredictionBatch(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, fov_data, roi_list, image_size=(60, 60), seqlen=None, z_channels=None):
        self.fov_data = fov_data
        self.roi_list = roi_list
        self.img_size = image_size
        self.z_channels = z_channels
        self.seqlen = 1 if seqlen is None else seqlen

    def __len__(self):
        """

        :return:
        """
        return len(self.fov_data) // self.seqlen + (len(self.fov_data) % self.seqlen != 0)

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        low = idx * self.seqlen
        high = min(low + self.seqlen, len(self.fov_data))
        roi_list = self.roi_list
        fov_data = self.fov_data.iloc[low:high]
        if self.seqlen > 1:
            roi_data = np.zeros((len(roi_list["roi"].unique()), min(self.seqlen, len(self.fov_data) - low), self.img_size[0],
                                 self.img_size[1], len(self.z_channels)))
        else:
            roi_data = np.zeros((len(roi_list["roi"].unique()), self.img_size[0], self.img_size[1], len(self.z_channels)))

        for row in fov_data.itertuples():
            rois = roi_list.loc[(roi_list['fov'] == row.fov) & (roi_list['t'] == row.t)]
            roi_ids = sorted(list(rois['roi'].unique()))

            # If merging and normalization are too slow, maybe use tensorflow or pytorch to do the operations
            fov_img = cv2.merge([cv2.imread(z_file, cv2.IMREAD_UNCHANGED) for z_file in reversed(row.channel_files)])

            for roi in rois.itertuples():
                roi_img = cv2.normalize(fov_img[roi.y0:roi.y1 + 1, roi.x0:roi.x1 + 1], dtype=cv2.CV_16F, dst=None, alpha=1e-10,
                                        beta=1.0, norm_type=cv2.NORM_MINMAX)
                if self.seqlen > 1:
                    roi_data[roi_ids.index(roi.roi), row.t % self.seqlen, ...] = tf.image.resize(np.array(roi_img), self.img_size,
                                                                                                 method='nearest')
                else:
                    roi_data[roi_ids.index(roi.roi), ...] = tf.image.resize(np.array(roi_img), self.img_size, method='nearest')
        return roi_data


class TblClassNamesRow(tbl.IsDescription):
    """

    """
    class_name = tbl.StringCol(16)


class Plugin(plugins.Plugin):
    """
    A class extending plugins.Plugin to handle the example plugin
    """
    id_ = 'gmgm.plewniak.roiclassification'
    version = '1.0.0'
    name = 'ROI classification'
    category = 'Deep learning'

    def __init__(self):
        super().__init__()
        self.menu = None
        # self.gui = None
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

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
                            items={'SGD'     : keras.optimizers.SGD,
                                   'Adam'    : keras.optimizers.Adam,
                                   'Adadelta': keras.optimizers.Adadelta,
                                   'Adamax'  : keras.optimizers.Adamax,
                                   'Nadam'   : keras.optimizers.Nadam,
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

    def register(self):
        """

        """
        # self.parameters.update()
        PyDetecDiv.app.project_selected.connect(self.update_parameters)
        PyDetecDiv.app.project_selected.connect(self.create_table)
        PyDetecDiv.app.viewer_roi_click.connect(self.add_context_action)

    def update_parameters(self, groups=None):
        """

        :param groups:
        """
        if groups in ['training']:
            self.parameters['weights'].clear()
        self.parameters.update(groups)
        self.parameters.reset(groups)

    def class_names(self, as_string=True):
        """
        return the classes

        :return: the class list
        """
        if as_string:
            return json.dumps(self.parameters['class_names'].value)
        return self.parameters['class_names'].value

    def create_table(self):
        """
        Create the table to save results if it does not exist yet
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            Base.metadata.create_all(project.repository.engine)

    def addActions(self, menu):
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

    def set_enabled_actions(self, manual_annotation, train_model, fine_tuning, predict, show_results, export_classification):
        """

        :param manual_annotation:
        :param train_model:
        :param fine_tuning:
        :param predict:
        :param show_results:
        :param export_classification:
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

    def add_context_action(self, data):
        """
        Add an action to annotate the ROI from the FOV viewer

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

    def import_annotated_rois(self):
        """
        Select a csv file containing ROI frames annotations and open a FOV2ROIlinks window to load the data it contains
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

    def manual_annotation(self, arg=None, roi_selection=None, run=None):
        """

        :param arg:
        :param roi_selection:
        :param run:
        """
        annotation_runs = self.get_annotation_runs()
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

    def resume_manual_annotation(self, annotator, roi_selection=None, run=None):
        """

        :param annotator:
        :param roi_selection:
        :param run:
        """
        # annotation_runs = self.get_annotation_runs()
        if annotator.parent() is None:
            tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / ROI annotation')
            tab.project_name = PyDetecDiv.project_name
            tab.set_top_tab(annotator, 'Manual annotation')
        if roi_selection:
            annotator.set_roi_list(roi_selection)
        if roi_selection is None:
            annotator.update_ROI_selection(self.class_names())
        else:
            annotator.set_roi_list(roi_selection)
            # annotator.next_roi()
        annotator.run = run
        annotator.setFocus()

    def show_results(self, arg=None, roi_selection=None):
        """

        :param arg:
        :param roi_selection:
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

    def get_classification_df(self, roi_selection=None, ground_truth=True, run_list=None):
        """

        :param roi_selection:
        :param ground_truth:
        :param run_list:
        :return:
        """
        annotation_runs = self.get_annotation_runs()
        if run_list is None:
            run_list = self.get_prediction_runs()[self.class_names()]
        ground_truth = {}
        predictions = {run: {} for run in run_list}
        for roi in self.get_annotated_rois(ids_only=False):
            if annotation_runs:
                ground_truth[roi.name] = self.get_classifications(roi=roi, run_list=annotation_runs[self.class_names()])

        for run in run_list:
            for roi in self.get_annotated_rois(run=run, ids_only=False):
                predictions[run][roi.name] = self.get_classifications(roi=roi, run_list=[run])

        if ground_truth:
            df = pd.DataFrame(
                    [[roi_name, frame, label] for roi_name in ground_truth for frame, label in enumerate(ground_truth[roi_name])],
                    columns=('roi', 'frame', 'ground truth'))

        for run in run_list:
            predictions_df = pd.DataFrame(
                    [[roi_name, frame, label] for roi_name in predictions[run] for frame, label in
                     enumerate(predictions[run][roi_name])],
                    columns=('roi', 'frame', f'run_{run}'))
            if ground_truth:
                df = df.merge(predictions_df, on=['roi', 'frame'], how='outer')
            else:
                df = predictions_df
        return df

    def export_classification_to_csv(self, trigger, filename='classification.csv', roi_selection=None, ground_truth=True,
                                     run_list=None):
        """

        :param trigger:
        :param filename:
        :param roi_selection:
        :param ground_truth:
        :param run_list:
        """
        print('export ROI classification')
        df = self.get_classification_df(roi_selection=roi_selection, ground_truth=ground_truth, run_list=run_list)
        # ok = {'frame': [frame for frame in df.loc[:,'frame'].unique()]}
        # if run_list is None:
        #     run_list = self.get_prediction_runs()[self.class_names()]
        # for run in run_list:
        #     ok[f'run_{run}'] = [df[df['frame'] == frame].apply(lambda row: (row[f'run_{run}']==row['ground truth']), axis=1).sum()
        #                         for frame in ok['frame']
        #                         ]
        # df = pd.DataFrame(ok)
        df.to_csv(os.path.join(get_project_dir(), filename))

    def get_classification_runs(self):
        """

        :return:
        """
        annotation_runs = self.get_annotation_runs()
        prediction_runs = self.get_prediction_runs()
        classification_runs = {classes: {'annotations': annotation_runs[classes], 'predictions': prediction_runs[classes]}
                               for classes in annotation_runs}
        return classification_runs

    def get_annotation_runs(self):
        """

        :return:
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

    def get_prediction_runs(self):
        """

        :return:
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = list(project.repository.session.execute(
                    sqlalchemy.text(f"SELECT run.id_,"
                                    f"run.parameters ->> '$.class_names' as class_names "
                                    f"FROM run "
                                    f"WHERE run.command='predict' "
                                    f"ORDER BY run.id_ ASC;")))
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

    def run_prediction(self):
        """

        """
        PredictionDialog(self)

    def run_training(self):
        """

        """
        if len(self.get_annotated_rois()) == 0:
            QMessageBox.critical(PyDetecDiv.main_window, 'No annotated ROI',
                                 'You should provide ground truth annotations for ROIs before training a model. '
                                 + 'Please, annotate ROIs or import annotations from a csv file.')
        else:
            TrainingDialog(self)

    def run_fine_tuning(self):
        """

        """
        self.update_parameters(groups='finetune')
        self.update_model_weights()
        if len(self.parameters['weights'].values) == 0:
            QMessageBox.critical(PyDetecDiv.main_window, 'No classifier to refine',
                                 'You need a trained model for fine-tuning. '
                                 + 'Please, train a model first or import a classifier from another project.')
        else:
            FineTuningDialog(self)

    def load_models(self):
        """
        Load available models (modules)

        """
        available_models = {}
        for _, name, _ in pkgutil.iter_modules(models.__path__):
            # self.parameters['model'].add_item(
            #     {name: importlib.import_module(f'.models.{name}', package=__package__)})
            available_models[name] = importlib.import_module(f'.models.{name}', package=__package__)
        for finder, name, _ in pkgutil.iter_modules([os.path.join(get_plugins_dir(), 'roi_classification/models')]):
            loader = finder.find_module(name)
            spec = importlib.util.spec_from_file_location(name, loader.path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            # self.parameters['model'].add_item({name: module})
            available_models[name] = module
        self.parameters['model'].set_items(available_models)

    def select_saved_parameters(self, weights_file):
        """

        :param weights_file:
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
                # self.parameters['num_test'].value = parameters['num_test']
                self.parameters['num_test'].value = 1.0 - parameters['num_training'] - parameters['num_validation']
                self.parameters['red_channel'].value = parameters['red_channel']
                self.parameters['green_channel'].value = parameters['green_channel']
                self.parameters['blue_channel'].value = parameters['blue_channel']
                self.parameters['dataset_seed'].value = parameters['dataset_seed']
                self.parameters['batch_size'].value = parameters['batch_size']
                self.parameters['seqlen'].value = parameters['seqlen']

    def update_model_weights(self, project_name=None):
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
            # try:
            #     all_users_path = os.path.join(get_plugins_dir(), 'roi_classification', 'models', parameters['model'])
            #     w_files.update({f: os.path.join(all_users_path, f) for f in os.listdir(all_users_path) if
            #                     os.path.isfile(os.path.join(all_users_path, f)) and f.endswith('.weights.h5')})
            # except FileNotFoundError:
            #     pass
        if w_files:
            self.parameters['weights'].set_items(w_files)

    def update_class_names(self, prediction=False):
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

    def update_channels(self):
        """

        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            image_resource = project.get_object('ImageResource', 1)
            n_layers = image_resource.zdim if image_resource else 0

        for param in ['red_channel', 'green_channel', 'blue_channel']:
            self.parameters[param].set_items({str(i): i for i in range(n_layers)})

    def update_fov_list(self):
        """

        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            self.parameters['fov'].set_items({fov.name: fov for fov in project.get_objects('FOV')})

    def save_annotations(self, roi, roi_classes, run):
        """
        Save manual annotation into the database

        :param roi: the annotated ROI
        :param roi_classes: the classes along time
        :param run: the annotation run
        """
        # with pydetecdiv_project(PyDetecDiv.project_name) as project:
        #     for t, class_name in enumerate(roi_classes):
        #         if class_name != '-':
        #             Results().save(project, run, roi, t, np.array([1]), [class_name])
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            for t, class_name in enumerate(roi_classes):
                if class_name != -1:
                    Results().save(project, run, roi, t, np.array([1]), [self.class_names(as_string=False)[class_name]])

    def get_annotated_rois(self, run=None, ids_only=False):
        """
        Get a list of annotated ROI frames

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
                            f"SELECT DISTINCT(roi) as annotated_rois FROM roi_classification, run "
                            f"WHERE run.id_=roi_classification.run "
                            f"AND (run.command='annotate_rois' OR run.command='import_annotated_rois') ",
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

    def get_unannotated_rois(self):
        """

        :return:
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            all_roi_ids = [roi.id_ for roi in project.get_objects('ROI')]
            print(f'All ROIs: {len(all_roi_ids)}')
            annotated_rois = self.get_annotated_rois(ids_only=True)
            print(f'Annotated ROIs: {len(annotated_rois)}')
            unannotated_roi_ids = set(all_roi_ids).difference(set(annotated_rois))
            print(f'Unannotated ROIs: {len(unannotated_roi_ids)}')
            return project.get_objects('ROI', list(unannotated_roi_ids)), project.get_objects('ROI', list(all_roi_ids))

    def get_annotation(self, roi, as_index=True):
        """
        Get the annotations for a ROI

        :param roi: the ROI
        :param as_index: bool set to True to return annotations as indices of class_names list, set to False to return
        annotations as class names
        :return: the list of annotated classes by frame
        """
        roi_classes = [-1] * roi.fov.image_resource().image_resource_data().sizeT
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = list(project.repository.session.execute(
                    sqlalchemy.text(f"SELECT rc.roi,rc.t,rc.class_name,"
                                    f"run.parameters ->> '$.annotator' as annotator, "
                                    f"run.parameters ->> '$.class_names' as class_names "
                                    f"FROM run, roi_classification as rc "
                                    f"WHERE (run.command='annotate_rois' OR run.command='import_annotated_rois') "
                                    f"AND rc.run=run.id_ and rc.roi={roi.id_} "
                                    f"AND annotator='{get_config_value('project', 'user')}' "
                                    f"AND run.parameters ->> '$.class_names'=json('{self.class_names()}') "
                                    f"ORDER BY rc.run ASC;")))
            if results:
                class_names = json.loads(results[0][4])
                if as_index:
                    for annotation in results:
                        roi_classes[annotation[1]] = class_names.index(annotation[2])
                else:
                    for annotation in results:
                        roi_classes[annotation[1]] = annotation[2]
        return roi_classes

    def get_classifications(self, roi, run_list, as_index=True):
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

    def get_all_annotations(self, z_layers=None):
        """

        :param z_layers:
        :return:
        """
        if z_layers is None:
            z_layers = (0,)

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = pd.DataFrame(project.repository.session.execute(
                    sqlalchemy.text(f"SELECT run.id_, rc.roi, roi.fov, roi.x0_, roi.y0_, roi.x1_, roi.y1_, "
                                    f"rc.t, data.c as channel, data.z, rc.class_name, data.url, img.key_val ->> '$.drift' as drift "
                                    f"FROM roi_classification as rc, ROI as roi, run, data, ImageResource as img "
                                    f"WHERE (run.command='annotate_rois' OR run.command='import_annotated_rois') "
                                    f"AND rc.run=run.id_ and rc.roi=roi.id_ "
                                    f"AND run.parameters ->> '$.annotator'='{get_config_value('project', 'user')}' "
                                    f"AND run.parameters ->> '$.class_names'=json('{self.class_names()}') "
                                    f"AND data.t = rc.t AND data.image_resource=img.id_ AND img.fov=roi.fov "
                                    f"AND data.z in {tuple(z_layers)} "
                                    f"ORDER BY rc.run, data.url, rc.roi ASC;")))
            return results

    def get_fov_data(self, z_layers=None, channel=None):
        """

        :param z_layers:
        :param channel:
        :return:
        """
        if z_layers is None:
            z_layers = (0,)
        if channel is None:
            channel = 0
            # channel = self.parameters['channel']

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = pd.DataFrame(project.repository.session.execute(
                    sqlalchemy.text(f"SELECT img.fov, data.t, data.url "
                                    f"FROM data, ImageResource as img "
                                    f"WHERE data.image_resource=img.id_ "
                                    f"AND data.z in {tuple(z_layers)} "
                                    f"AND data.c={channel} "
                                    f"ORDER BY img.fov, data.url ASC;")))
            fov_data = results.groupby(['fov', 't'])['url'].apply(self.layers2channels).reset_index()
        fov_data.columns = ['fov', 't', 'channel_files']
        return fov_data

    def get_roi_list(self):
        """

        :return:
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = pd.DataFrame(project.repository.session.execute(
                    sqlalchemy.text(f"SELECT id_ as roi, fov, x0_ as x0, y0_ as y0, x1_ as x1, y1_ as y1 "
                                    f"FROM ROI "
                                    f"ORDER BY fov, id_ ASC;")))
        return results

    def get_annotations(self):
        """

        :return:
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

    def get_drift_corrections(self):
        """

        :return:
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            results = pd.DataFrame(project.repository.session.execute(
                    sqlalchemy.text(f"SELECT fov, img.key_val ->> '$.drift' as drift "
                                    f"FROM ImageResource as img "
                                    f"ORDER BY fov ASC;")))
            drift_corrections = pd.DataFrame(columns=['fov', 't', 'dx', 'dy'])
            for row in results.itertuples(index=False):
                if row.drift is not None:
                    df = pd.read_csv(os.path.join(get_project_dir(), row.drift))
                    df['fov'] = row.fov
                    df['t'] = df.index
                    drift_corrections = pd.concat([drift_corrections, df], ignore_index=True)
            return drift_corrections

    def layers2channels(self, zfiles):
        """

        :param zfiles:
        :return:
        """
        zfiles = list(zfiles)
        return [zfiles[i] for i in [self.parameters['red_channel'].value, self.parameters['green_channel'].value,
                                    self.parameters['blue_channel'].value]]

    def create_hdf5_annotated_rois(self, hdf5_file, z_channels=None, channel=0):
        """

        :param hdf5_file:
        :param z_channels:
        :param channel:
        """
        print(f'{datetime.now().strftime("%H:%M:%S")}: Retrieving data for annotated ROIs')
        data = self.get_annotations()
        print(f'{datetime.now().strftime("%H:%M:%S")}: Data retrieved with {len(data)} rows')
        class_names = self.class_names(as_string=False)

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

    def prepare_data_for_training(self, hdf5_file, seqlen=0, train=0.6, validation=0.2, seed=42):
        """

        :param hdf5_file:
        :param seqlen:
        :param train:
        :param validation:
        :param seed:
        :return:
        """
        h5file = tbl.open_file(hdf5_file, mode='r')
        roi_data = h5file.root.roi_data
        print(f'{datetime.now().strftime("%H:%M:%S")}: Reading targets into a numpy array')
        targets_arr = h5file.root.targets
        num_frames = targets_arr.shape[0]
        num_rois = targets_arr.shape[1]
        targets = targets_arr.read()
        print(f'{datetime.now().strftime("%H:%M:%S")}: Select valid targets from array with shape {targets.shape}')
        if seqlen == 0:
            indices = [[frame, roi] for roi in range(num_rois) for frame in range(num_frames) if targets[frame, roi] != -1]
        else:
            indices = [[frame, roi] for roi in range(num_rois) for frame in range(0, num_frames - seqlen + 1, seqlen)
                       if np.all([targets[frame:frame + seqlen, roi] != -1])]
        print(f'{datetime.now().strftime("%H:%M:%S")}: Kept {len(indices)} valid ROI frames or sequences')

        print(f'{datetime.now().strftime("%H:%M:%S")}: Shuffling data')
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        print(f'{datetime.now().strftime("%H:%M:%S")}: Determine training and validation datasets size')
        num_training = int(len(indices) * train)
        num_validation = int(len(indices) * validation)
        print(f'{datetime.now().strftime("%H:%M:%S")}: Close HDF5 file and return datasets indices')
        h5file.close()
        return indices[:num_training], indices[num_training:num_validation + num_training], indices[num_validation + num_training:]

    def prepare_data_for_classification(self, fov_list, z_channels=None):
        """

        :param fov_list:
        :param z_channels:
        :return:
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

    def compute_class_weights(self) -> list:
        """

        """
        class_counts = dict(
                Counter([x for roi in self.get_annotated_rois() for x in self.get_annotation(roi) if x >= 0]))
        n = len(class_counts)
        total = sum([v for c, v in class_counts.items()])
        weights = {k: total / (n * class_counts[k]) for k in class_counts.keys()}
        for k in range(n):
            if k not in weights:
                weights[k] = 0.00
        return weights

    def lr_decay(self, epoch, lr):
        """
        Learning rate scheduler

        :param epoch: the current epoch
        :param lr: the current learning rate
        :return: the new learning rate
        """
        if (epoch != 0) & (epoch % self.parameters['decay_period'].value == 0):
            return lr * self.parameters['decay_rate'].value
        return lr

    def train_model(self):
        """
        Launch training a model: select the network, load weights (optional), define the training, validation
        and test sets, then run the training using training and validation sets and the evaluation on the test set.
        """
        log_dir = os.path.join(get_project_dir(), 'logs', 'fit', datetime.now().strftime("%Y%m%d-%H%M%S"))
        tf.keras.utils.set_random_seed(self.parameters['seed'].value)
        batch_size = self.parameters['batch_size'].value
        epochs = self.parameters['epochs'].value
        z_channels = [self.parameters['red_channel'].value, self.parameters['green_channel'].value,
                      self.parameters['blue_channel'].value]

        module = self.parameters['model'].value
        print(module.__name__)

        model = module.model.create_model(len(self.parameters['class_names'].value))

        print(f'{datetime.now().strftime("%H:%M:%S")}: Compiling model')
        learning_rate = self.parameters['learning_rate'].value
        if self.parameters['optimizer'].key in ['SGD']:
            optimizer = self.parameters['optimizer'].value(learning_rate=learning_rate,
                                                           momentum=self.parameters['momentum'].value)
        else:
            optimizer = self.parameters['optimizer'].value(learning_rate=learning_rate)
        lr_metric = get_lr_metric(optimizer)

        model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy', lr_metric],
                )
        # model.summary(print_fn=lambda x: print(x, file=sys.stderr))
        # print('model output:', model.layers[-1].output.shape, file=sys.stderr)
        input_shape = model.layers[0].output.shape
        image_shape = (input_shape[-3], input_shape[-2])

        if len(input_shape) == 5:
            seqlen = self.parameters['seqlen'].value
            print(f'{datetime.now().strftime("%H:%M:%S")}: Sequence length: {seqlen}\n')
        else:
            seqlen = 0

        # hdf5_file = os.path.join(get_project_dir(), 'data', 'annotated_rois.h5')
        os.makedirs(os.path.join(get_project_dir(), 'roi_classification', 'data'), exist_ok=True)
        hdf5_file = os.path.join(get_project_dir(), 'roi_classification', 'data', 'annotated_rois.h5')
        print(hdf5_file)
        if not os.path.exists(hdf5_file):
            self.create_hdf5_annotated_rois(hdf5_file, z_channels=z_channels)

        print(f'{datetime.now().strftime("%H:%M:%S")}: Preparing data for training')
        training_idx, validation_idx, test_idx = self.prepare_data_for_training(hdf5_file, seqlen=seqlen,
                                                                                train=self.parameters[
                                                                                    'num_training'].value,
                                                                                validation=self.parameters[
                                                                                    'num_validation'].value,
                                                                                seed=self.parameters[
                                                                                    'dataset_seed'].value)

        print(f'{datetime.now().strftime("%H:%M:%S")}: Training dataset (size: {len(training_idx)})')
        training_dataset = DataProvider(hdf5_file, training_idx, image_shape=image_shape, seqlen=seqlen, batch_size=batch_size,
                                        name='training', targets=True)

        print(f'{datetime.now().strftime("%H:%M:%S")}: Validation dataset (size: {len(validation_idx)})')
        validation_dataset = DataProvider(hdf5_file, validation_idx, image_shape=image_shape, seqlen=seqlen, batch_size=batch_size,
                                          name='validation', targets=True)

        print(f'{datetime.now().strftime("%H:%M:%S")}: Test dataset (size: {len(test_idx)})')
        test_dataset = DataProvider(hdf5_file, test_idx, image_shape=image_shape, seqlen=seqlen, batch_size=batch_size,
                                    name='test', targets=True, shuffle=False)

        if self.parameters['weights'].value is not None:
            print(f'{datetime.now().strftime("%H:%M:%S")}: Loading weights from {self.parameters["weights"].key}')
            # loadWeights(model, filename=self.parameters['weights'].value)
            model.load_weights(self.parameters['weights'].value)
            run = self.save_training_run(finetune=True)
        else:
            run = self.save_training_run()

        self.save_training_datasets(hdf5_file, training_idx, validation_idx, test_idx)

        checkpoint_monitor_metric = self.parameters['checkpoint_metric'].value
        best_checkpoint_filename = f'{run.id_}_best_{checkpoint_monitor_metric}.weights.h5'
        checkpoint_filepath = os.path.join(get_project_dir(), 'roi_classification', 'models',
                                           self.parameters['model'].key,
                                           f'{best_checkpoint_filename}')

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor=checkpoint_monitor_metric,
                mode='auto',
                verbose=1,
                save_best_only=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                              profile_batch=(0, 20))

        training_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', start_from_epoch=1, min_delta=0,
                                                                   patience=5, verbose=1, mode='auto', baseline=None,
                                                                   restore_best_weights=True)

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(self.lr_decay, verbose=0)

        callbacks = [model_checkpoint_callback, learning_rate_scheduler, tensorboard_callback]

        if self.parameters['early_stopping'].value:
            callbacks += [training_early_stopping]

        print(f'{datetime.now().strftime("%H:%M:%S")}: Starting training')
        history = model.fit(training_dataset, epochs=epochs, callbacks=callbacks, validation_data=validation_dataset,
                            verbose=2)

        last_weights_filename = f'{run.id_}_last.weights.h5'
        model.save_weights(os.path.join(get_project_dir(), 'roi_classification', 'models',
                                        self.parameters['model'].key, last_weights_filename), overwrite=True)

        run.parameters.update({'last_weights': last_weights_filename, 'best_weights': best_checkpoint_filename})
        run.validate().commit()

        print(f'{datetime.now().strftime("%H:%M:%S")}: Evaluation on test dataset')
        evaluation = dict(zip(model.metrics_names, model.evaluate(test_dataset, verbose=2)))

        ground_truth = [label for batch in [y for x, y in test_dataset] for label in batch]
        if len(input_shape) == 4:
            print(f'{datetime.now().strftime("%H:%M:%S")}: Prediction on test dataset for last model')
            predictions = model.predict(test_dataset, verbose=2).argmax(axis=1)

            print(f'{datetime.now().strftime("%H:%M:%S")}: Prediction on test dataset for best model')
            model.load_weights(checkpoint_filepath)
            # model.compile(optimizer=optimizer,
            #               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            #               metrics=['accuracy', lr_metric], )
            best_predictions = model.predict(test_dataset).argmax(axis=1)
        else:
            print(f'{datetime.now().strftime("%H:%M:%S")}: Prediction on test dataset for last model')
            predictions = [label for seq in model.predict(test_dataset, verbose=2).argmax(axis=2) for label in seq]

            print(f'{datetime.now().strftime("%H:%M:%S")}: Prediction on test dataset for best model')
            model.load_weights(checkpoint_filepath)
            # model.compile(optimizer=optimizer,
            #               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            #               metrics=['accuracy', lr_metric], )
            best_predictions = [label for seq in model.predict(test_dataset, verbose=2).argmax(axis=2) for label in seq]
            ground_truth = [label for seq in ground_truth for label in seq]

        labels = list(range(len(self.parameters['class_names'].value)))
        precision, recall, fscore, support = precision_recall_fscore_support(ground_truth, predictions, labels=labels,
                                                                             zero_division=np.nan)
        stats = {'last_stats': {'precision': list(precision),
                                'recall'   : list(recall),
                                'fscore'   : list(fscore),
                                'support'  : [int(s) for s in support]
                                }
                 }
        if run.key_val is None:
            run.key_val = stats
        else:
            run.key_val.update(stats)

        run.validate().commit()

        rows = [
            f'| {self.parameters["class_names"].value[i]} | {precision[i]:.2f} | {recall[i]:.2f} | {fscore[i]:.2f} | {support[i]}'
            for i in labels]
        stats = '\n'.join(rows)
        # print('**Last epoch model evaluation on test set**')
        print(f"""
        
        
**Last epoch model evaluation on test set**

| class | precision | recall | F-score | support 
|:------|----------:|-------:|--------:|--------:
{stats}

""")

        precision, recall, fscore, support = precision_recall_fscore_support(ground_truth, best_predictions, labels=labels,
                                                                             zero_division=np.nan)

        stats = {'best_stats': {'precision': list(precision),
                                'recall'   : list(recall),
                                'fscore'   : list(fscore),
                                'support'  : [int(s) for s in support]
                                }
                 }
        if run.key_val is None:
            run.key_val = stats
        else:
            run.key_val.update(stats)

        run.validate().commit()

        rows = [
            f'| {self.parameters["class_names"].value[i]} | {precision[i]:.2f} | {recall[i]:.2f} | {fscore[i]:.2f} | {support[i]}'
            for i in labels]
        stats = '\n'.join(rows)
        # print('**Best model evaluation on test set**')
        print(f"""
**Best model evaluation on test set**

| class | precision | recall | F-score | support 
|:------|----------:|-------:|--------:|--------:
{stats}

""")

        if self.parameters['weights'].value is not None:
            self.update_model_weights()

        # tf.profiler.experimental.stop()
        training_dataset.close()
        validation_dataset.close()
        test_dataset.close()

        return (module.__name__, self.parameters['class_names'].value, history, evaluation, ground_truth, predictions,
                best_predictions)

    def save_training_run(self, finetune=False):
        """
        save the current training Run

        :param seqlen: the sequence length
        :param epochs: the number of epochs
        :param batch_size: the batch size
        :param module: the module name (i.e. the network that was trained)
        :return: the current Run instance
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            if finetune:
                return self.save_run(project, 'fine_tune', self.parameters.json(groups='finetune'))
            return self.save_run(project, 'train_model', self.parameters.json(groups='training'))

    def save_training_datasets(self, hdf5_file, training_idx, validation_idx, test_idx):
        training_ds = Dataset(project=self.run.project, name=f'train_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                              type_='training', run=self.run.id_)
        validation_ds = Dataset(project=self.run.project, name=f'val_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                                type_='validation', run=self.run.id_)
        test_ds = Dataset(project=self.run.project, name=f'test_{datetime.now().strftime("%Y%m%d-%H%M%S")}', type_='test',
                          run=self.run.id_)

        # print(len(training_idx), len(validation_idx), len(test_idx))

        h5file = tbl.open_file(hdf5_file, mode='r')
        targets = h5file.root.targets.read()
        h5file.close()

        for frame, roi_id in training_idx:
            TrainingData().save(self.run.project, roi_id, frame, targets[frame, roi_id], training_ds.id_)

        for frame, roi_id in validation_idx:
            TrainingData().save(self.run.project, roi_id, frame, targets[frame, roi_id], validation_ds.id_)

        for frame, roi_id in test_idx:
            TrainingData().save(self.run.project, roi_id, frame, targets[frame, roi_id], test_ds.id_)

        self.run.project.commit()

    def predict(self):
        """
        Running prediction on all ROIs in selected FOVs.
        """
        seqlen = self.parameters['seqlen'].value
        z_channels = [self.parameters['red_channel'].value, self.parameters['green_channel'].value,
                      self.parameters['blue_channel'].value]
        fov_names = [self.parameters['fov'].key]

        module = self.parameters['model'].value
        print(module.__name__)
        model = module.model.create_model(len(self.parameters['class_names'].value))
        print(f'{datetime.now().strftime("%H:%M:%S")}: Loading weights')
        weights = self.parameters['weights'].value
        if weights:
            # loadWeights(model, filename=self.parameters['weights'].value)
            model.load_weights(self.parameters['weights'].value)

        input_shape = model.layers[0].output.shape

        print(f'{datetime.now().strftime("%H:%M:%S")}: Compiling model')
        model.compile()

        print(f'{datetime.now().strftime("%H:%M:%S")}: Preparing data')
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            fov_ids = [fov.id_ for fov in [project.get_named_object('FOV', fov_name) for fov_name in fov_names]]

        fov_data, roi_list, rois = self.prepare_data_for_classification(fov_ids, z_channels)
        num_rois = len(rois)

        # print(roi_list.loc[:, ['t', 'roi']].to_numpy(), file=sys.stderr)

        if len(input_shape) == 4:
            img_size = (input_shape[1], input_shape[2])
            roi_dataset = PredictionBatch(fov_data, roi_list, image_size=img_size, z_channels=z_channels)
        else:
            img_size = (input_shape[2], input_shape[3])
            roi_dataset = PredictionBatch(fov_data, roi_list, image_size=img_size, seqlen=seqlen, z_channels=z_channels)

        print(f'{datetime.now().strftime("%H:%M:%S")}: Making predictions')

        predictions = model.predict(roi_dataset, verbose=2)
        # print(predictions.shape, file=sys.stderr)

        if len(input_shape) > 4:
            if not isinstance(predictions, tf.RaggedTensor):
                predictions = tf.RaggedTensor.from_tensor(predictions)
            grouped_tensors = []
            for roi_idx in range(num_rois):
                # Collect rows for this ROI: i, i + num_rois, i + 2*num_rois, ...
                indices = tf.range(roi_idx, predictions.shape[0], delta=num_rois)
                grouped_rows = tf.gather(predictions, indices)  # Gather ragged slices
                concatenated_rows = tf.concat(grouped_rows, axis=0)  # Concatenate along the time dimension
                grouped_tensors.append(concatenated_rows.flat_values)

            # 2. Stack results into a dense tensor
            predictions = tf.stack(grouped_tensors, axis=0).numpy()
            # print(predictions.shape, file=sys.stderr)

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            print(f'{datetime.now().strftime("%H:%M:%S")}: Saving results')
            parameters = self.parameters.json(groups='prediction')
            run = self.save_run(project, 'predict', parameters)
            if len(input_shape) > 4:
                for roi_id, prediction in zip(rois, predictions):
                    for frame, p in enumerate(prediction):
                        # print(f'ROI: {roi_id}, t:{frame}, {p}', file=sys.stderr)
                        Results().save(project, run, project.get_object('ROI', roi_id), frame, p, self.class_names(as_string=False))
            else:
                for (roi_id, frame), p in zip(roi_list.loc[:, ['roi', 't']].to_numpy(), predictions):
                    # print(f'ROI: {roi_id}, t:{frame}, {prediction}', file=sys.stderr)
                    Results().save(project, run, project.get_object('ROI', roi_id), frame, p, self.class_names(as_string=False))
        print(f'{datetime.now().strftime("%H:%M:%S")}: predictions OK')

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
                # run.command in ['train_model', 'fine_tune', 'import_classifier']]
                for run in run_list:
                    run.parameters['project'] = project_name
                    run.parameters['run'] = run.id_
                    run.command = 'import_classifier'
                    self.classifiers.add_item(
                            {f"{project_name}-{run.id_} {run.parameters['model']} {run.parameters['class_names']}": run})
        with pydetecdiv_project(current_project_name) as project:  # resetting global project name
            pass
        ImportClassifierDialog(self)

    def import_classifier(self) -> None:
        """
        Imports a classifier, i.e. a combination of a deep-learning network/model, weights trained on annotated data and
        class names
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            run: Run = self.classifiers.value
            user_path = str(os.path.join(get_project_dir(), 'roi_classification', 'models', run.parameters['model']))
            os.makedirs(user_path, exist_ok=True)
            origin_path = str(os.path.join(get_project_dir(run.parameters['project']), 'roi_classification', 'models',
                                           run.parameters['model']))
            copy_files([os.path.join(origin_path, run.parameters['best_weights']),
                        os.path.join(origin_path, run.parameters['last_weights'])], user_path)
            new_run = Run(project=project, **(run.record(no_id=True)))

        print(f'Classifier imported from project {run.parameters["project"]}')

    def draw_annotated_rois(self):
        """
        Draw annotated ROIs as rectangles coloured according to the class
        """
        colours = [
            QColor(255, 0, 0, 64),
            QColor(0, 255, 0, 64),
            QColor(0, 0, 255, 64),
            QColor(255, 255, 0, 64),
            QColor(0, 255, 255, 64),
            QColor(255, 0, 255, 64),
            QColor(64, 128, 0, 64),
            QColor(64, 0, 128, 64),
            QColor(128, 64, 0, 64),
            QColor(0, 64, 128, 64),
            ]
        rec_items = {item.data(0): item for item in PyDetecDiv.main_window.active_subwindow.viewer.scene.items() if
                     isinstance(item, QGraphicsRectItem)}
        for roi in self.get_annotated_rois():
            if roi.name in rec_items:
                annotation = self.get_annotation(roi)[PyDetecDiv.main_window.active_subwindow.viewer.T]
                rec_items[roi.name].setBrush(colours[annotation])


def get_lr_metric(optimizer):
    """
    Get the learning rate metric for optimizer for use during training to monitor the learning rate

    :param optimizer: the optimizer
    :return: the learning rate function
    """

    def lr(y_true, y_pred):
        return optimizer.learning_rate

    return lr


def lr_exp_decay(epoch, lr):
    """
    Learning rate scheduler for exponential decay

    :param epoch: the current epoch
    :param lr: the current learning rate
    :return: the new learning rate
    """
    k = 0.1
    if epoch == 0:
        return lr
    return lr * math.exp(-k)
