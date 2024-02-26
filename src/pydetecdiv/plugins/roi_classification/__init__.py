"""
An example plugin showing how to interact with database
"""
import importlib
import json
import math
import os.path
import pkgutil
import random
import sys
from datetime import datetime

import h5py
import numpy as np
import sqlalchemy
from PySide6.QtGui import QAction, QColor, QPen
from PySide6.QtSql import QSqlDatabase, QSqlQuery
from PySide6.QtWidgets import QGraphicsRectItem, QFileDialog
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import registry
from sqlalchemy.types import JSON
import tensorflow as tf

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, get_plugins_dir
from pydetecdiv.domain.Image import Image, ImgDType

from .gui import ROIclassification, FOV2ROIlinks
from . import models
from .gui.annotate import open_annotator
from ...app.gui.Windows import MatplotViewer
from ...domain.Dataset import Dataset
from ...settings import get_config_value

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


def prepare_data(data_list, seqlen=None, targets=True):
    roi_data_list = []
    for roi in data_list:
        imgdata = roi.fov.image_resource().image_resource_data()
        seqlen = seqlen if seqlen else 1
        if targets:
            annotation_indices = get_annotation(roi)
            for i in range(0, imgdata.sizeT, seqlen):
                sequence = annotation_indices[i:i + seqlen]
                if len(sequence) == seqlen and all(a > 0 for a in sequence):
                    roi_data_list.extend([ROIdata(roi, imgdata, sequence, i)])
        else:
            roi_data_list.extend([ROIdata(roi, imgdata, None, frame) for frame in range(0, imgdata.sizeT, seqlen)])
    return roi_data_list


def get_annotation(roi):
    roi_classes = [-1] * roi.fov.image_resource().image_resource_data().sizeT
    with pydetecdiv_project(PyDetecDiv().project_name) as project:
        results = list(project.repository.session.execute(
            sqlalchemy.text(f"SELECT rc.roi,rc.t,rc.class_name,"
                            f"run.parameters ->> '$.annotator' as annotator, "
                            f"run.parameters ->> '$.class_names' as class_names "
                            f"FROM run, roi_classification as rc "
                            f"WHERE run.command='annotate_rois' and rc.run=run.id_ and rc.roi={roi.id_} "
                            f"AND annotator='{get_config_value('project', 'user')}' "
                            f"ORDER BY rc.run ASC;")))
        class_names = json.loads(results[0][4])
        for annotation in results:
            roi_classes[annotation[1]] = class_names.index(annotation[2])
    return roi_classes


class ROIdata:
    def __init__(self, roi, imgdata, target=None, frame=0):
        self.roi = roi
        self.imgdata = imgdata
        self.target = target
        self.frame = frame


class ROIDataset(tf.keras.utils.Sequence):
    def __init__(self, roi_data_list, image_size=(60, 60), class_names=None, batch_size=32, seqlen=None,
                 z_channels=None):
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
        self.class_names = []
        self.gui = None

    def create_table(self):
        """
        Create the table to save results if it does not exist yet
        """
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
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
        PyDetecDiv().viewer_roi_click.connect(self.add_context_action)

    def add_context_action(self, data):
        """
        Add an action to annotate the ROI from the FOV viewer
        :param data: the data sent by the PyDetecDiv().viewer_roi_click signal
        """
        if self.gui:
            r, menu = data
            with pydetecdiv_project(PyDetecDiv().project_name) as project:
                selected_roi = project.get_named_object('ROI', r.data(0))
                if selected_roi:
                    roi_list = [selected_roi]
                    annotate = menu.addAction('Annotate region class')
                    annotate.triggered.connect(lambda _: open_annotator(self, roi_list))

    def load_model(self):
        module = self.gui.network.currentData()
        print(module.__name__)
        # model = module.load_model(load_weights=False)
        model = module.model.create_model()
        print('Loading weights')
        weights = self.gui.weights.currentData()
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

        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            print('Saving run')
            run = self.save_run(project, 'predict', {'fov': fov_names,
                                                     'network': self.gui.network.currentData().__name__,
                                                     'weights': self.gui.weights.currentData(),
                                                     'class_names': self.class_names,
                                                     'red': self.gui.red_channel.currentIndex(),
                                                     'green': self.gui.green_channel.currentIndex(),
                                                     'blue': self.gui.blue_channel.currentIndex()
                                                     })
            roi_list = np.ndarray.flatten(np.array([roi for roi in [fov.roi_list for fov in
                                                                    [project.get_named_object('FOV', fov_name) for
                                                                     fov_name in
                                                                     fov_names]]]))

            if len(input_shape) == 4:
                img_size = (input_shape[1], input_shape[2])
                roi_data_list = prepare_data(roi_list, targets=False)
                roi_dataset = ROIDataset(roi_data_list, image_size=img_size, class_names=self.class_names,
                                         batch_size=batch_size, z_channels=z_channels)
            else:
                img_size = (input_shape[2], input_shape[3])
                roi_data_list = prepare_data(roi_list, seqlen, targets=False)
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
            self.gui = ROIclassification(PyDetecDiv().main_window)
            self.load_models(self.gui)
            self.gui.update_model_weights()
            self.gui.action_menu.addItem('Create new model', userData=self.create_model)
            self.gui.action_menu.addItem('Annotate ROIs', userData=self.annotate_rois)
            self.gui.action_menu.addItem('Train model', userData=self.train_model)
            self.gui.action_menu.addItem('Classify ROIs', userData=self.predict)
            self.update_class_names()
            self.set_table_view(PyDetecDiv().project_name)
            self.set_sequence_length(PyDetecDiv().project_name)
            PyDetecDiv().project_selected.connect(self.set_table_view)
            PyDetecDiv().project_selected.connect(self.set_sequence_length)
            PyDetecDiv().project_selected.connect(self.create_table)
            PyDetecDiv().saved_rois.connect(self.set_table_view)
            self.gui.roi_import_box.accepted.connect(self.import_annotated_rois)
            # PyDetecDiv().main_window.active_subwindow.viewer.video_frame.connect(self.draw_annotated_rois)
            self.gui.button_box.accepted.connect(self.run)
            self.gui.network.currentIndexChanged.connect(self.update_class_names)
            self.gui.action_menu.currentIndexChanged.connect(self.adapt_gui)
            self.gui.action_menu.setCurrentIndex(3)
        self.gui.setVisible(True)
        # self.draw_annotated_rois()

    def adapt_gui(self):
        """
        Modify the appearance of the GUI according to the selected action
        """
        match (self.gui.action_menu.currentIndex()):
            case 0:
                # Create new model
                self.gui.roi_selection.hide()
                self.gui.roi_sample.hide()
                self.gui.roi_import.hide()
                self.gui.classifier_selectionLayout.setRowVisible(1, False)
                self.gui.preprocessing.show()
                self.gui.misc_box.hide()
                self.gui.network.setEditable(True)
                self.gui.classes.setReadOnly(False)
                self.gui.datasets.hide()
            case 1:
                # Annotate ROIs
                self.gui.roi_selection.hide()
                self.gui.roi_sample.show()
                self.gui.roi_import.show()
                self.gui.classifier_selectionLayout.setRowVisible(1, False)
                self.gui.preprocessing.hide()
                self.gui.misc_box.hide()
                self.gui.network.setEditable(False)
                self.gui.classes.setReadOnly(True)
                self.gui.datasets.hide()
            case 2:
                # Train model
                self.gui.roi_selection.hide()
                self.gui.roi_sample.hide()
                self.gui.roi_import.hide()
                self.gui.classifier_selectionLayout.setRowVisible(1, True)
                self.gui.preprocessing.show()
                self.gui.misc_box.show()
                self.gui.misc_boxLayout.setRowVisible(self.gui.epochs, True)
                self.gui.network.setEditable(False)
                self.gui.classes.setReadOnly(True)
                self.gui.datasets.show()
            case 3:
                # Classify ROIs
                self.gui.roi_selection.show()
                self.gui.roi_sample.hide()
                self.gui.roi_import.hide()
                self.gui.classifier_selectionLayout.setRowVisible(1, True)
                self.gui.preprocessing.show()
                self.gui.misc_box.show()
                self.gui.misc_boxLayout.setRowVisible(self.gui.epochs, False)
                self.gui.network.setEditable(False)
                self.gui.classes.setReadOnly(True)
                self.gui.datasets.hide()
            case _:
                pass
        self.gui.resize(self.gui.form.sizeHint())

    def annotate_rois(self):
        """
        Launch the annotator GUI for ROI annotation
        """
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            selected_rois = random.sample(project.get_objects('ROI'), self.gui.roi_number.value())
        open_annotator(self, selected_rois)

    def save_annotations(self, roi, roi_classes, run):
        """
        Save manual annotation into the database
        :param roi: the annotated ROI
        :param roi_classes: the classes along time
        :param run: the annotation run
        """
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            for t, class_name in enumerate(roi_classes):
                if class_name != '-':
                    Results().save(project, run, roi, t, np.array([1]), [class_name])

    def update_class_names(self):
        """
        Gets the names of classes from the GUI
        """
        if self.gui:
            self.class_names = self.gui.network.currentData().class_names

    def set_table_view(self, project_name):
        """
        Set the content of the Table view to display the available ROIs to classify
        :param project_name: the name of the project
        """
        if project_name:
            with pydetecdiv_project(project_name) as project:
                self.gui.update_list(project)

    def set_sequence_length(self, project_name):
        """
        Set the maximum value for sequence length
        :param project_name: the name of the project
        """
        if project_name:
            with pydetecdiv_project(project_name) as project:
                self.gui.update_sequence_length(project)

    def create_model(self):
        """
        Launch model creation.
        """
        print('Not implemented')

    def train_model(self):
        """
        Launch training a model: select the network, load weights (optional), define the training, validation
        and test sets, then run the training using training and validation sets and the evaluation on the test set.
        """
        batch_size = self.gui.batch_size.value()
        seqlen = self.gui.seq_length.value()
        epochs = self.gui.epochs.value()
        z_channels = [self.gui.red_channel.currentIndex(), self.gui.green_channel.currentIndex(),
                      self.gui.blue_channel.currentIndex()]

        module = self.gui.network.currentData()
        print(module.__name__)
        # model = module.load_model(load_weights=False)
        model = module.model.create_model()
        print('Loading weights')
        weights = self.gui.weights.currentData()
        if weights:
            loadWeights(model, filename=self.gui.weights.currentData())

        print('Compiling model')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        input_shape = model.layers[0].output.shape
        # print(model.layers[-1].output.shape)

        if len(input_shape) == 4:
            img_size = (input_shape[1], input_shape[2])

            roi_list = prepare_data(get_annotated_rois())
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

            roi_list = prepare_data(get_annotated_rois(), seqlen)
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

        run_id = self.save_training_run(roi_list, seqlen, num_training, num_validation, epochs, batch_size)

        checkpoint_filepath = os.path.join(get_plugins_dir(), 'roi_classification', 'models',
                                           self.gui.network.currentText(),
                                           f'weights_{PyDetecDiv().project_name}_{run_id}_best.h5')

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        histories = {'Training': model.fit(training_dataset, epochs=epochs,
                                           # steps_per_epoch=num_training, #//batch_size,
                                           callbacks=[model_checkpoint_callback],
                                           validation_data=validation_dataset,
                                           verbose=2,
                                           # workers=4, use_multiprocessing=True
                                           )}
        model.save(os.path.join(get_plugins_dir(), 'roi_classification', 'models',
                                self.gui.network.currentText(),
                                f'weights_{PyDetecDiv().project_name}_{run_id}_last.h5'), overwrite=True,
                   save_format='h5')
        # print(histories)
        tab = PyDetecDiv().main_window.add_tabbed_window(f'{PyDetecDiv().project_name} / {module.__name__}')
        tab.viewer.project_name = PyDetecDiv().project_name
        history_plot = plot_history(histories)
        tab.addTab(history_plot, 'Training')
        tab.setCurrentWidget(history_plot)

    def save_training_run(self, roi_list, seqlen, num_training, num_validation, epochs, batch_size, ):
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            run = self.save_run(project, 'train_model', {'class_names': self.class_names,
                                                         'seqlen': seqlen,
                                                         'num_training': self.gui.training_data.value(),
                                                         'num_validation': self.gui.validation_data.value(),
                                                         'batch_size': batch_size,
                                                         'epochs': epochs,
                                                         })
            training_ds = Dataset(project=project, name=f'train_{datetime.now().strftime("%Y%m%d-%H%M")}',
                                  type_='training', run=run.id_)
            validation_ds = Dataset(project=project, name=f'val_{datetime.now().strftime("%Y%m%d-%H%M")}',
                                    type_='validation', run=run.id_)
            test_ds = Dataset(project=project, name=f'test_{datetime.now().strftime("%Y%m%d-%H%M")}', type_='test',
                              run=run.id_)

            print(num_training, num_validation)
            for data in roi_list[:num_training]:
                TrainingData().save(project, data.roi, data.frame, data.target, training_ds.id_)

            for data in roi_list[num_training:num_training + num_validation]:
                TrainingData().save(project, data.roi, data.frame, data.target, validation_ds.id_)

            for data in roi_list[num_training + num_validation:]:
                TrainingData().save(project, data.roi, data.frame, data.target, test_ds.id_)
        return run.id_

    def import_annotated_rois(self):
        filters = ["csv (*.csv)", ]
        annotation_file, _ = QFileDialog.getOpenFileName(self.gui, caption='Choose file with annotated ROIs',
                                                         dir='.',
                                                         filter=";;".join(filters),
                                                         selectedFilter=filters[0])
        FOV2ROIlinks(annotation_file, self)

    def save_results(self, project, run, roi, frame, class_name):
        Results().save(project, run, roi, frame, np.array([1]), [class_name])


def plot_history(history):
    """
    Plots metrics histories: accuracy and loss for training and fine tuning.
    :param history: list of metrics histories to plot
    """
    plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, columns=2, rows=len(history))
    # fig, axs = plt.subplots(2, len(history))
    axs = plot_viewer.axes
    # fig.suptitle('Model accuracy')
    if len(history) == 1:
        for k in history:
            axs[0].plot(history[k].history['accuracy'])
            axs[0].plot(history[k].history['val_accuracy'])
            axs[0].set_ylabel('accuracy')
            axs[0].set_xlabel('epoch')
            axs[0].legend(['train', 'val'], loc='lower right')
            axs[1].plot(history[k].history['loss'])
            axs[1].plot(history[k].history['val_loss'])
            axs[1].legend(['train', 'val'], loc='upper right')
            axs[1].set_ylabel('loss')
    else:
        i = 0
        for k in history:
            axs[0, i].plot(history[k].history['accuracy'])
            axs[0, i].plot(history[k].history['val_accuracy'])
            axs[0, i].set_title(k)
            axs[0, i].set_ylabel('accuracy')
            axs[0, i].set_xlabel('epoch')
            axs[0, i].legend(['train', 'val'], loc='lower right')
            axs[1, i].plot(history[k].history['loss'])
            axs[1, i].plot(history[k].history['val_loss'])
            axs[1, i].legend(['train', 'val'], loc='upper right')
            axs[1, i].set_ylabel('loss')
            i += 1
    plot_viewer.canvas.draw()
    return plot_viewer


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
                                                           drift=None)).stretch_contrast(),
                                Image(imgdata.image_memmap(sliceX=slice(roi.x, roi.x + roi.width),
                                                           sliceY=slice(roi.y, roi.y + roi.height),
                                                           C=0, Z=z[1], T=t,
                                                           drift=None)).stretch_contrast(),
                                Image(imgdata.image_memmap(sliceX=slice(roi.x, roi.x + roi.width),
                                                           sliceY=slice(roi.y, roi.y + roi.height),
                                                           C=0, Z=z[2], T=t,
                                                           drift=None)).stretch_contrast(),
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
        # z = [self.gui.red_channel.currentIndex(),
        #      self.gui.green_channel.currentIndex(),
        #      self.gui.blue_channel.currentIndex()]
    image1 = Image(imgdata.image(T=t, Z=z[0]))
    image2 = Image(imgdata.image(T=t, Z=z[1]))
    image3 = Image(imgdata.image(T=t, Z=z[2]))

    # print(f'Composing for frame {t}')
    roi_images = [Image.compose_channels([image1.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast(),
                                          image2.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast(),
                                          image3.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast()
                                          ]).as_tensor(ImgDType.float32) for roi in roi_list]
    return roi_images


def draw_annotated_rois():
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
    rec_items = {item.data(0): item for item in PyDetecDiv().main_window.active_subwindow.viewer.scene.items() if
                 isinstance(item, QGraphicsRectItem)}
    for roi in get_annotated_rois():
        if roi.name in rec_items:
            annotation = get_annotation(roi)[PyDetecDiv().main_window.active_subwindow.viewer.T]
            rec_items[roi.name].setBrush(colours[annotation])


def get_annotated_rois():
    with pydetecdiv_project(PyDetecDiv().project_name) as project:
        db = QSqlDatabase("QSQLITE")
        db.setDatabaseName(project.repository.name)
        db.open()
        query = QSqlQuery(
            "SELECT DISTINCT(roi) as annotated_rois FROM roi_classification, run "
            "WHERE run.id_=roi_classification.run "
            "AND run.command='annotate_rois';",
            db=db)
        query.exec()
        if query.first():
            roi_ids = [query.value('annotated_rois')]
            while query.next():
                roi_ids.append(query.value('annotated_rois'))
            return project.get_objects('ROI', roi_ids)
        return []


def display_dataset(dataset, sequences=False):
    for dset in dataset.__iter__():
        ds = dset[0] if isinstance(dset, tuple) else dset
        for data in ds:
            tab = PyDetecDiv().main_window.add_tabbed_window('Showing dataset')
            if sequences is False:
                plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, columns=1, rows=1)
            else:
                plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, columns=len(data), rows=1)
            axs = plot_viewer.axes
            tab.addTab(plot_viewer, 'training dataset')
            if sequences is False:
                axs.imshow(data)
            else:
                for i, img in enumerate(data):
                    axs[i].imshow(img)
        plot_viewer.canvas.draw()


def loadWeights(model, filename=os.path.join(__path__[0], "weights.h5"), debug=False):
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
    # Returns the index to the layer
    layers = model.layers
    for i in range(len(layers)):
        if layerName == layers[i].name:
            return i
    print("")
    print("WEIGHT LOADING FAILED. MODEL DOES NOT CONTAIN LAYER WITH NAME: ", layerName)
    print("")
    return -1


def intList(myList):
    # Converts a list of numbers into a list of ints.
    return list(map(int, myList))
