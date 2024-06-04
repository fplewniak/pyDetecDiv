"""
An plugin for deep-learning ROI classification
"""
import importlib
import json
import math
import os.path
import pkgutil
import random
import sys
from collections import Counter
from datetime import datetime

import h5py
import numpy as np
import sqlalchemy
from PySide6.QtGui import QAction, QColor
from PySide6.QtSql import QSqlDatabase, QSqlQuery
from PySide6.QtWidgets import QGraphicsRectItem
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import registry
from sqlalchemy.types import JSON
import tensorflow as tf

from sklearn.metrics import ConfusionMatrixDisplay

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, get_project_dir
from pydetecdiv.settings import get_plugins_dir
from pydetecdiv.domain import Image, Dataset, ImgDType
from pydetecdiv.settings import get_config_value

from .gui import FOV2ROIlinks, ROIclassificationDialog
from . import models
from .gui.annotate import open_annotator
from ...app.gui.core.widgets.viewers.plots import MatplotViewer

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

    def save(self, project, roi, t, target, dataset):
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
        self.t = t
        self.target = target
        self.dataset = dataset
        project.repository.session.add(self)


class ROIdata:
    """
    ROI data, linking ROI object, the corresponding image data, target (class), and frame
    """

    def __init__(self, roi, imgdata, target=None, frame=0):
        self.roi = roi
        self.imgdata = imgdata
        self.target = target
        self.frame = frame


class ROIDataset(tf.keras.utils.Sequence):
    """
    ROI dataset that can be used to feed the model for training, evaluation or prediction
    """

    def __init__(self, roi_data_list, image_size=(60, 60), class_names=None, batch_size=32, seqlen=None,
                 z_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.img_size = image_size
        self.class_names = class_names
        self.batch_size = batch_size
        self.roi_data_list = roi_data_list
        self.seqlen = seqlen
        self.z_channels = z_channels

    def __len__(self):
        return math.ceil(len(self.roi_data_list) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.roi_data_list))
        batch_roi = self.roi_data_list[low:high]
        batch_targets = []
        batch_data = []
        for data in batch_roi:
            if self.seqlen is None:
                roi_dataset = get_rgb_images_from_stacks_memmap(imgdata=data.imgdata, roi_list=[data.roi], t=data.frame,
                                                                z=self.z_channels)
                if data.target is not None:
                    batch_targets.append(data.target[0])
            else:
                roi_dataset = get_images_sequences(imgdata=data.imgdata, roi_list=[data.roi], t=data.frame,
                                                   seqlen=self.seqlen, z=self.z_channels)
                if data.target is not None:
                    batch_targets.append(data.target)
            img_array = tf.convert_to_tensor([tf.image.resize(i, self.img_size, method='nearest') for i in roi_dataset])
            batch_data.append(img_array[0])
        if batch_targets:
            return np.array(batch_data), np.array(batch_targets)
        return np.array(batch_data)


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
        self.gui = None

    @property
    def class_names(self):
        """
        return the classes

        :return: the class list
        """
        return json.loads(self.gui.classes.text())

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
        self.menu = menu
        action_launch = QAction("ROI classification", self.menu)
        action_launch.triggered.connect(self.launch)
        self.menu.addAction(action_launch)
        PyDetecDiv.app.viewer_roi_click.connect(self.add_context_action)

    def add_context_action(self, data):
        """
        Add an action to annotate the ROI from the FOV viewer

        :param data: the data sent by the PyDetecDiv().viewer_roi_click signal
        """
        if self.gui:
            r, menu = data
            with pydetecdiv_project(PyDetecDiv.project_name) as project:
                selected_roi = project.get_named_object('ROI', r.data(0))
                if selected_roi:
                    roi_list = [selected_roi]
                    annotate = menu.addAction('Annotate region class')
                    annotate.triggered.connect(lambda _: open_annotator(self, roi_list))

    def load_model(self):
        """
        Load the model

        :return: the model
        """
        module = self.gui.network.currentData()
        print(module.__name__)
        model = module.model.create_model(len(self.class_names))
        print('Loading weights')
        weights = self.gui.weights.currentData()
        print(weights)
        if weights:
            loadWeights(model, filename=self.gui.weights.currentData())

        print('Compiling model')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    def predict(self):
        """
        Running prediction on all available ROIs.
        """
        model = self.load_model()
        input_shape = model.layers[0].output.shape
        batch_size = self.gui.batch_size.value()
        seqlen = self.gui.seq_length.value()
        fov_names = [index.data() for index in self.gui.selection_model.selectedRows(0)]
        z_channels = [self.gui.red_channel.currentIndex(), self.gui.green_channel.currentIndex(),
                      self.gui.blue_channel.currentIndex()]

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            print('Saving run')
            parameters = {'fov': fov_names}
            parameters.update(self.parameters.get_values('classify'))
            run = self.save_run(project, 'predict', parameters)
            # run = self.save_run(project, 'predict', {'fov': fov_names,
            #                                          'network': self.gui.network.currentData().__name__,
            #                                          'weights': self.gui.weights.currentData(),
            #                                          'class_names': self.class_names,
            #                                          'red': self.gui.red_channel.currentIndex(),
            #                                          'green': self.gui.green_channel.currentIndex(),
            #                                          'blue': self.gui.blue_channel.currentIndex()
            #                                          })
            # roi_list = np.ndarray.flatten(np.array([roi for roi in [fov.roi_list for fov in
            #                                                         [project.get_named_object('FOV', fov_name) for
            #                                                          fov_name in
            #                                                          fov_names]]]))
            roi_list = np.ndarray.flatten(np.array(list([fov.roi_list for fov in
                                                         [project.get_named_object('FOV', fov_name) for
                                                          fov_name in
                                                          fov_names]])))

            if len(input_shape) == 4:
                img_size = (input_shape[1], input_shape[2])
                roi_data_list = self.prepare_data(roi_list, targets=False)
                roi_dataset = ROIDataset(roi_data_list, image_size=img_size, class_names=self.class_names,
                                         batch_size=batch_size, z_channels=z_channels)
            else:
                img_size = (input_shape[2], input_shape[3])
                roi_data_list = self.prepare_data(roi_list, seqlen, targets=False)
                roi_dataset = ROIDataset(roi_data_list, image_size=img_size, class_names=self.class_names,
                                         seqlen=seqlen, batch_size=batch_size, z_channels=z_channels)

            predictions = model.predict(roi_dataset)
            # display_dataset(roi_dataset, sequences=len(input_shape) != 4)

            for (prediction, data) in zip(np.squeeze(predictions), roi_data_list):
                if len(input_shape) == 4:
                    # max_score, max_index = max((value, index) for index, value in enumerate(prediction))
                    # print(data.roi.name, data.frame, self.class_names[max_index], max_score)
                    Results().save(project, run, data.roi, data.frame, prediction, self.class_names)
                else:
                    for i in range(seqlen):
                        # max_score, max_index = max((value, index) for index, value in enumerate(prediction[i]))
                        # print(data.roi.name, data.frame + i, self.class_names[max_index], max_score)
                        if (data.frame + i) < data.imgdata.sizeT:
                            Results().save(project, run, data.roi, data.frame + i, prediction[i], self.class_names)
        print('predictions OK')

    def load_models(self, gui):
        """
        Load available models (modules)

        :param gui: the GUI
        """
        for _, name, _ in pkgutil.iter_modules(models.__path__):
            gui.network.addItem(name, userData=importlib.import_module(f'.models.{name}', package=__package__))
        for finder, name, _ in pkgutil.iter_modules([os.path.join(get_plugins_dir(), 'roi_classification/models')]):
            loader = finder.find_module(name)
            spec = importlib.util.spec_from_file_location(name, loader.path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            gui.network.addItem(name, userData=module)
        gui.update()

    def run(self):
        """
        Run the action selected in the GUI (create new model, annotate ROIs, train model, classify ROIs)
        """
        self.gui.action_menu.currentData()()

    def launch(self):
        """
        Display the ROI classification docked GUI window
        """
        if self.gui is None:
            self.create_table()
            PyDetecDiv.app.project_selected.connect(self.create_table)
            self.gui = ROIclassificationDialog(self, title='ROI class prediction (Deep Learning)')
            self.gui.update_all()
        self.gui.setVisible(True)

    def annotate_rois(self):
        """
        Launch the annotator GUI for ROI annotation
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            selected_rois = random.sample(project.get_objects('ROI'), self.gui.roi_number.value())
        open_annotator(self, selected_rois)

    def save_annotations(self, roi, roi_classes, run):
        """
        Save manual annotation into the database

        :param roi: the annotated ROI
        :param roi_classes: the classes along time
        :param run: the annotation run
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            for t, class_name in enumerate(roi_classes):
                if class_name != '-':
                    Results().save(project, run, roi, t, np.array([1]), [class_name])

    def get_annotated_rois(self):
        """
        Get a list of annotated ROI frames

        :return: the list of annotated ROI frames
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            db = QSqlDatabase("QSQLITE")
            db.setDatabaseName(project.repository.name)
            db.open()
            query = QSqlQuery(
                f"SELECT DISTINCT(roi) as annotated_rois FROM roi_classification, run "
                f"WHERE run.id_=roi_classification.run "
                f"AND (run.command='annotate_rois' OR run.command='import_annotated_rois') "
                f"AND run.parameters ->> '$.class_names'='{self.gui.classes.text()}' ;",
                db=db)
            query.exec()
            if query.first():
                roi_ids = [query.value('annotated_rois')]
                while query.next():
                    roi_ids.append(query.value('annotated_rois'))
                return project.get_objects('ROI', roi_ids)
            return []

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
                                f"AND run.parameters ->> '$.class_names'='{self.gui.classes.text()}' "
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

    def prepare_data(self, data_list, seqlen=None, targets=True):
        """
        Prepare the data from a list of ROI object as a list of ROIData objects to build the ROIDataset instance

        :param data_list: the ROI list
        :param seqlen: the length of the frame sequence
        :param targets: should targets be included in the dataset or not
        :return: the ROIData list
        """
        roi_data_list = []
        for roi in data_list:
            imgdata = roi.fov.image_resource().image_resource_data()
            seqlen = seqlen if seqlen else 1
            if targets:
                annotation_indices = self.get_annotation(roi)
                for i in range(0, imgdata.sizeT, seqlen):
                    sequence = annotation_indices[i:i + seqlen]
                    if len(sequence) == seqlen and all(a >= 0 for a in sequence):
                        roi_data_list.extend([ROIdata(roi, imgdata, sequence, i)])
            else:
                roi_data_list.extend([ROIdata(roi, imgdata, None, frame) for frame in range(0, imgdata.sizeT, seqlen)])
        return roi_data_list

    def compute_class_weights(self):
        class_counts = dict(
            Counter([x for roi in self.get_annotated_rois() for x in self.get_annotation(roi) if x >= 0]))
        print(class_counts)
        n = len(class_counts)
        total = sum([v for c, v in class_counts.items()])
        weights = {k: total / (n * class_counts[k]) for k in class_counts.keys()}
        for k in range(n):
            if k not in weights:
                weights[k] = 0.00
        return weights

    def create_model(self):
        """
        Launch model creation.
        """
        print('Not implemented')

    def lr_decay(self, epoch, lr):
        """
        Learning rate scheduler

        :param epoch: the current epoch
        :param lr: the current learning rate
        :return: the new learning rate
        """
        if (epoch != 0) & (epoch % self.gui.decay_freq.value() == 0):
            return lr * self.gui.decay_rate.value()
        return lr

    def train_model(self):
        """
        Launch training a model: select the network, load weights (optional), define the training, validation
        and test sets, then run the training using training and validation sets and the evaluation on the test set.
        """
        tf.keras.utils.set_random_seed(self.gui.weight_seed.value())
        batch_size = self.gui.batch_size.value()
        seqlen = self.gui.seq_length.value()
        epochs = self.gui.epochs.value()
        z_channels = [self.gui.red_channel.currentIndex(), self.gui.green_channel.currentIndex(),
                      self.gui.blue_channel.currentIndex()]

        module = self.gui.network.currentData()
        print(module.__name__)

        run = self.save_training_run(module)

        model = module.model.create_model(len(self.class_names))
        print('Loading weights')
        weights = self.gui.weights.currentData()
        if weights:
            loadWeights(model, filename=self.gui.weights.currentData())

        print('Compiling model')
        learning_rate = self.gui.learning_rate.value()
        if self.gui.optimizer.currentText() in ['SGD']:
            optimizer = self.gui.optimizer.currentData()(learning_rate=learning_rate,
                                                         momentum=self.gui.momentum.value())
        else:
            optimizer = self.gui.optimizer.currentData()(learning_rate=learning_rate)
        lr_metric = get_lr_metric(optimizer)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy', lr_metric],
        )
        # print(model.summary())
        input_shape = model.layers[0].output.shape
        # print(input_shape)
        # print(seqlen)
        # print(model.layers[-1].output.shape)

        if len(input_shape) == 4:
            img_size = (input_shape[1], input_shape[2])

            roi_list = self.prepare_data(self.get_annotated_rois())
            random.seed(self.gui.datasets_seed.value())
            random.shuffle(roi_list)
            num_training = int(self.gui.training_data.value() * len(roi_list))
            num_validation = int(self.gui.validation_data.value() * len(roi_list))

            print('Training dataset')
            training_dataset = ROIDataset(roi_list[:num_training], z_channels=z_channels,
                                          image_size=img_size, class_names=self.class_names, batch_size=batch_size)
            print('Validation dataset')
            validation_dataset = ROIDataset(roi_list[num_training:num_training + num_validation], z_channels=z_channels,
                                            image_size=img_size, class_names=self.class_names, batch_size=batch_size)
            print('Test dataset')
            test_dataset = ROIDataset(roi_list[num_training + num_validation:], z_channels=z_channels,
                                      image_size=img_size,
                                      class_names=self.class_names, batch_size=batch_size)
        else:
            img_size = (input_shape[2], input_shape[3])

            roi_list = self.prepare_data(self.get_annotated_rois(), seqlen)
            random.seed(self.gui.datasets_seed.value())
            random.shuffle(roi_list)
            num_training = round(self.gui.training_data.value() * len(roi_list))
            num_validation = round(self.gui.validation_data.value() * len(roi_list))

            print('Training dataset')
            training_dataset = ROIDataset(roi_list[:num_training], image_size=img_size, class_names=self.class_names,
                                          seqlen=seqlen, batch_size=batch_size, z_channels=z_channels, )
            print('Validation dataset')
            validation_dataset = ROIDataset(roi_list[num_training:num_training + num_validation],
                                            image_size=img_size, class_names=self.class_names, seqlen=seqlen,
                                            batch_size=batch_size, z_channels=z_channels, )
            print('Test dataset')
            test_dataset = ROIDataset(roi_list[num_training + num_validation:], z_channels=z_channels,
                                      image_size=img_size,
                                      class_names=self.class_names, seqlen=seqlen, batch_size=batch_size)

        # display_dataset(training_dataset, sequences=len(input_shape) != 4)

        self.save_training_datasets(run, roi_list, num_training, num_validation)

        checkpoint_monitor_metric = self.gui.checkpoint_metric.currentData()
        best_checkpoint_filename = f'{run.id_}_best_{checkpoint_monitor_metric}.weights.h5'
        checkpoint_filepath = os.path.join(get_project_dir(), 'roi_classification', 'models',
                                           self.gui.network.currentText(),
                                           f'{best_checkpoint_filename}')

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor=checkpoint_monitor_metric,
            mode='auto',
            verbose=1,
            save_best_only=True)

        training_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', start_from_epoch=1, min_delta=0,
                                                                   patience=5, verbose=1, mode='auto', baseline=None,
                                                                   restore_best_weights=True)

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(self.lr_decay, verbose=0)

        callbacks = [model_checkpoint_callback, learning_rate_scheduler]

        if self.gui.early_stopping.isChecked():
            callbacks += [training_early_stopping]

        # class_weights = compute_class_weights() if self.gui.class_weights.isChecked() else {k: 1.0 for k in range(len(self.class_names))}

        history = model.fit(training_dataset, epochs=epochs,
                            callbacks=callbacks, validation_data=validation_dataset, verbose=2, )

        last_weights_filename = f'{run.id_}_last.weights.h5'
        model.save_weights(os.path.join(get_project_dir(), 'roi_classification', 'models',
                                        self.gui.network.currentText(), last_weights_filename), overwrite=True)

        run.parameters.update({'last_weights': last_weights_filename, 'best_weights': best_checkpoint_filename})
        run.validate().commit()

        # evaluation = {metrics: value for metrics, value in zip(model.metrics_names, model.evaluate(test_dataset))}
        evaluation = dict(zip(model.metrics_names, model.evaluate(test_dataset)))

        ground_truth = [label for batch in [y for x, y in test_dataset] for label in batch]
        if len(input_shape) == 4:
            predictions = model.predict(test_dataset).argmax(axis=1)
            model.load_weights(checkpoint_filepath)
            model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy', lr_metric], )
            best_predictions = model.predict(test_dataset).argmax(axis=1)
        else:
            predictions = [label for seq in model.predict(test_dataset).argmax(axis=2) for label in seq]
            model.load_weights(checkpoint_filepath)
            model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy', lr_metric], )
            best_predictions = [label for seq in model.predict(test_dataset).argmax(axis=2) for label in seq]
            ground_truth = [label for seq in ground_truth for label in seq]

        tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / {module.__name__}')
        tab.project_name = PyDetecDiv.project_name
        history_plot = plot_history(history, evaluation)
        tab.addTab(history_plot, 'Training')
        tab.setCurrentWidget(history_plot)

        confusion_matrix_plot = plot_confusion_matrix(ground_truth, predictions, self.class_names)
        tab.addTab(confusion_matrix_plot, 'Confusion matrix (last epoch)')

        confusion_matrix_plot = plot_confusion_matrix(ground_truth, best_predictions, self.class_names)
        tab.addTab(confusion_matrix_plot, 'Confusion matrix (best checkpoint)')

        self.gui.update_model_weights()

    def save_training_run(self, module):
        """
        save the current training Run

        :param seqlen: the sequence length
        :param epochs: the number of epochs
        :param batch_size: the batch size
        :param module: the module name (i.e. the network that was trained)
        :return: the current Run instance
        """
        parameters = {'model': module.__name__}
        parameters.update(self.parameters.get_values('training'))
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            return self.save_run(project, 'train_model', parameters)

    def save_training_datasets(self, run, roi_list, num_training, num_validation):
        """
        save the datasets used for training and evaluation in the database

        :param run: the current run
        :param roi_list: the list of ROI/frames
        :param num_training: the number of training data
        :param num_validation: the number of validation data
        """
        project = roi_list[0].roi.project
        training_ds = Dataset(project=project, name=f'train_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                              type_='training', run=run.id_)
        validation_ds = Dataset(project=project, name=f'val_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                                type_='validation', run=run.id_)
        test_ds = Dataset(project=project, name=f'test_{datetime.now().strftime("%Y%m%d-%H%M%S")}', type_='test',
                          run=run.id_)

        print(num_training, num_validation)
        for data in roi_list[:num_training]:
            TrainingData().save(project, data.roi, data.frame, data.target, training_ds.id_)

        for data in roi_list[num_training:num_training + num_validation]:
            TrainingData().save(project, data.roi, data.frame, data.target, validation_ds.id_)

        for data in roi_list[num_training + num_validation:]:
            TrainingData().save(project, data.roi, data.frame, data.target, test_ds.id_)
        project.commit()

    def save_results(self, project, run, roi, frame, class_name):
        """
        Save the results in database

        :param project: the current project
        :param run: the current run
        :param roi: the current ROI
        :param frame: the current frame
        :param class_name: the class name
        """
        Results().save(project, run, roi, frame, np.array([1]), [class_name])

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


def plot_history(history, evaluation):
    """
    Plots metrics history.

    :param history: metrics history to plot
    :param evaluation: metrics from model evaluation on test dataset, shown as horizontal dashed lines on the plots
    """
    plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=2, rows=1)
    axs = plot_viewer.axes
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].axhline(evaluation['accuracy'], color='red', linestyle='--')
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'val'], loc='lower right')
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].axhline(evaluation['loss'], color='red', linestyle='--')
    axs[1].legend(['train', 'val'], loc='upper right')
    axs[1].set_ylabel('loss')

    plot_viewer.show()
    return plot_viewer


def plot_confusion_matrix(ground_truth, predictions, class_names):
    """
    Plot the confusion matrix normalized i) by rows (recall in diagonals) and ii) by columns (precision in diagonals)

    :param ground_truth: the ground truth index values
    :param predictions: the predicted index values
    :param class_names: the class names
    :return: the plot viewer where the confusion matrix is plotted
    """
    plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=2, rows=1)
    plot_viewer.axes[0].set_title('Normalized by row')
    ConfusionMatrixDisplay.from_predictions(ground_truth, predictions, labels=list(range(len(class_names))),
                                            display_labels=class_names, normalize='true', ax=plot_viewer.axes[0])
    plot_viewer.axes[1].set_title('Normalized by column')
    ConfusionMatrixDisplay.from_predictions(ground_truth, predictions, labels=list(range(len(class_names))),
                                            display_labels=class_names, normalize='pred', ax=plot_viewer.axes[1])
    return plot_viewer


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


def get_images_sequences(imgdata, roi_list, t, seqlen=None, z=None):
    """
    Get a sequence of seqlen images for each roi

    :param imgdata: the image data resource
    :param roi_list: the list of ROIs
    :param t: the starting time point (index of frame)
    :param seqlen: the number of frames
    :return: a tensor containing the sequences for all ROIs
    """
    maxt = min(imgdata.sizeT, t + seqlen) if seqlen else imgdata.sizeT
    roi_sequences = tf.stack(
        [get_rgb_images_from_stacks_memmap(imgdata, roi_list, f, z=z) for f in range(t, maxt)],
        axis=1)
    if roi_sequences.shape[1] < seqlen:
        padding_config = [[0, 0], [seqlen - roi_sequences.shape[1], 0], [0, 0], [0, 0], [0, 0]]
        roi_sequences = tf.pad(roi_sequences, padding_config, mode='CONSTANT', constant_values=0.0)
    # print('roi sequence', roi_sequences.shape)
    return roi_sequences


def get_rgb_images_from_stacks_memmap(imgdata, roi_list, t, z=None):
    """
    Combine 3 z-layers of a grayscale image resource into a RGB image where each of the z-layer is a channel

    :param imgdata: the image data resource
    :param roi_list: the list of ROIs
    :param t: the frame index
    :param z: a list of 3 z-layer indices defining the grayscale layers that must be combined as channels
    :return: a tensor of the combined RGB images
    """
    if z is None:
        z = [0, 0, 0]
    roi_images = [
        Image.compose_channels([Image(imgdata.image_memmap(sliceX=slice(roi.x, roi.x + roi.width),
                                                           sliceY=slice(roi.y, roi.y + roi.height),
                                                           C=0, Z=z[0], T=t,
                                                           drift=PyDetecDiv.app.apply_drift)).stretch_contrast(),
                                Image(imgdata.image_memmap(sliceX=slice(roi.x, roi.x + roi.width),
                                                           sliceY=slice(roi.y, roi.y + roi.height),
                                                           C=0, Z=z[1], T=t,
                                                           drift=PyDetecDiv.app.apply_drift)).stretch_contrast(),
                                Image(imgdata.image_memmap(sliceX=slice(roi.x, roi.x + roi.width),
                                                           sliceY=slice(roi.y, roi.y + roi.height),
                                                           C=0, Z=z[2], T=t,
                                                           drift=PyDetecDiv.app.apply_drift)).stretch_contrast(),
                                ]).as_tensor(ImgDType.float32) for roi in roi_list]
    return roi_images


def get_rgb_images_from_stacks(imgdata, roi_list, t, z=None):
    """
    Combine 3 z-layers of a grayscale image resource into a RGB image where each of the z-layer is a channel

    :param imgdata: the image data resource
    :param roi_list: the list of ROIs
    :param t: the frame index
    :param z: a list of 3 z-layer indices defining the grayscale layers that must be combined as channels
    :return: a tensor of the combined RGB images
    """
    if z is None:
        z = [0, 0, 0]

    image1 = Image(imgdata.image(T=t, Z=z[0], drift=PyDetecDiv.app.apply_drift))
    image2 = Image(imgdata.image(T=t, Z=z[1], drift=PyDetecDiv.app.apply_drift))
    image3 = Image(imgdata.image(T=t, Z=z[2], drift=PyDetecDiv.app.apply_drift))

    roi_images = [Image.compose_channels([image1.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast(),
                                          image2.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast(),
                                          image3.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast()
                                          ]).as_tensor(ImgDType.float32) for roi in roi_list]
    return roi_images


def display_dataset(dataset, sequences=False):
    """
    Display a dataset in plot viewer

    :param dataset: the dataset to display
    :param sequences: whether or not to show frame sequences
    """
    for dset in dataset.__iter__():
        ds = dset[0] if isinstance(dset, tuple) else dset
        for data in ds:
            tab = PyDetecDiv.main_window.add_tabbed_window('Showing dataset')
            if sequences is False:
                plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=1, rows=1)
            else:
                plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=len(data), rows=1)
            axs = plot_viewer.axes
            tab.addTab(plot_viewer, 'training dataset')
            if sequences is False:
                axs.imshow(data)
            else:
                for i, img in enumerate(data):
                    axs[i].imshow(img)
        plot_viewer.show()


def loadWeights(model, filename=os.path.join(__path__[0], "weights.h5"), debug=False):
    """
    load the weights into the model

    :param model: the model
    :param filename: the H5 file name containing the weights
    :param debug: debug mode
    """
    with h5py.File(filename, 'r') as f:
        if 'backend' in f.attrs:
            # Keras-saved model weights, cannot be loaded as below
            model.load_weights(filename)
        else:
            # Every layer is an h5 group. Ignore non-groups (such as /0)
            for g in f:
                if isinstance(f[g], h5py.Group):
                    group = f[g]
                    layerName = group.attrs['Name']
                    numVars = int(group.attrs['NumVars'])
                    if debug:
                        print("layerName:", layerName)
                        print("    numVars:", numVars)
                    # Find the layer index from its namevar
                    layerIdx = layerNum(model, layerName)
                    layer = model.layers[layerIdx]
                    if debug:
                        print("    layerIdx=", layerIdx)
                    # Every weight is an h5 dataset in the layer group. Read the weights
                    # into a list in the correct order
                    weightList = [0] * numVars
                    for d in group:
                        dataset = group[d]
                        varName = dataset.attrs['Name']
                        shp = intList(dataset.attrs['Shape'])
                        weightNum = int(dataset.attrs['WeightNum'])
                        # Read the weight and put it into the right position in the list
                        if debug:
                            print("    varName:", varName)
                            print("        shp:", shp)
                            print("        weightNum:", weightNum)
                        weightList[weightNum] = tf.constant(dataset[()], shape=shp)
                    # Assign the weights into the layer
                    for w in range(numVars):
                        if debug:
                            print("Copying variable of shape:")
                            print(weightList[w].shape)
                        layer.variables[w].assign(weightList[w])
                        if debug:
                            print("Assignment successful.")
                            print("Set variable value:")
                            print(layer.variables[w])
                    # Finalize layer state
                    if hasattr(layer, 'finalize_state'):
                        layer.finalize_state()


def layerNum(model, layerName):
    """
    Returns the index to the layer

    :param model: the model
    :param layerName: the name of the layer
    :return: the index of the layer
    """
    layers = model.layers
    for i in range(len(layers)):
        if layerName == layers[i].name:
            return i
    print("")
    print("WEIGHT LOADING FAILED. MODEL DOES NOT CONTAIN LAYER WITH NAME: ", layerName)
    print("")
    return -1


def intList(myList):
    """
    Converts a list of numbers into a list of ints.

    :param myList: the list to be converted
    :return: the converted list
    """
    return list(map(int, myList))
