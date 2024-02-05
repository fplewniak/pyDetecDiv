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

import numpy as np
import psutil
import sqlalchemy
from PySide6.QtGui import QAction, QColor, QPen
from PySide6.QtSql import QSqlDatabase, QSqlQuery
from PySide6.QtWidgets import QGraphicsRectItem
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import registry
from sqlalchemy.types import JSON
import tensorflow as tf

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, get_plugins_dir
from pydetecdiv.domain.Image import Image, ImgDType
from pydetecdiv.utils import split_list

from .gui import ROIclassification
from . import models
from .gui.annotate import open_annotator
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


def prepare_data(data_list, seqlen):
    roi_data_list = []
    for roi in data_list:
        imgdata = roi.fov.image_resource().image_resource_data()
        # annotation_indices = [0] * roi.fov.image_resource().sizeT
        annotation_indices = split_list(get_annotation(roi), -1, seqlen)
        roi_data_list.extend([ROIdata(roi, imgdata, annotation_seq) for annotation_seq in annotation_indices])
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


# class DatasetGenerator:
#     def __init__(self, data_list, plugin, timeseries=True):
#         self.data_list = data_list
#         self.timeseries = timeseries
#         self.plugin = plugin
#
#     def __iter__(self):
#         for data in self.data_list:
#             if self.timeseries:
#                 roi_sequences = self.plugin.get_images_sequences(data.imgdata, [data.roi], 0,
#                                                           seqlen=self.plugin.gui.seq_length.value())
#                 img_array = tf.convert_to_tensor(
#                     [tf.image.resize(i, (224, 224), method='nearest') for i in roi_sequences])
#                 print(data.roi.name)
#                 yield (img_array, np.array([data.target]))
#             else:
#                 print('not implemented')

class ROIdata:
    def __init__(self, roi, imgdata, target=None):
        self.roi = roi
        self.imgdata = imgdata
        self.target = target


class ROIDataset(tf.keras.utils.Sequence):
    def __init__(self, roi_list, image_size=(60, 60), class_names=None, batch_size=32, seqlen=None):
        self.roi_list = roi_list
        self.img_size = image_size
        self.class_names = class_names
        self.batch_size = batch_size
        # self.roi_data_list = self.prepare_data(roi_list)
        self.roi_data_list = roi_list
        self.seqlen = seqlen

    def __len__(self):
        return math.ceil(len(self.roi_list) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.roi_list))
        batch_roi = self.roi_data_list[low:high]
        batch_targets = []
        batch_data = []
        for data in batch_roi:
            if self.seqlen is None:
                roi_dataset = get_rgb_images_from_stacks_memmap(imgdata=data.imgdata, roi_list=[data.roi], t=0)
                batch_targets.append(data.target[0])
            else:
                roi_dataset = get_images_sequences(imgdata=data.imgdata, roi_list=[data.roi], t=0,
                                                          seqlen=self.seqlen)
                batch_targets.append(data.target[0:self.seqlen])
            img_array = tf.convert_to_tensor([tf.image.resize(i, self.img_size, method='nearest') for i in roi_dataset])
            batch_data.append(img_array[0])
        # print(
        #     f'{np.format_float_positional(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), precision=1)} MB')
        return np.array(batch_data), np.array(batch_targets)


class Plugin(plugins.Plugin):
    """
    A class extending plugins.Plugin to handle the example plugin
    """
    id_ = 'gmgm.plewniak.roiclassification'
    version = '1.0.0'
    name = 'ROI classification'
    category = 'Deep learning'
    # class_names = []

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
        # self.menu = menu.addMenu(self.name)
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

    def predict(self):
        """
        Method launching the plugin. This may encapsulate (as it is the case here) the call to a GUI or some domain
        functionalities run directly without any further interface.
        """
        module = self.gui.network.currentData()
        print(module.__name__)
        model = module.load_model(load_weights=False)
        print('Loading weights')
        weights = self.gui.weights.currentData()
        if weights:
            module.loadWeights(model, filename=self.gui.weights.currentData())

        print('Compiling model')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        input_shape = model.layers[0].output.shape
        batch_size = self.gui.batch_size.value()
        fov_names = [index.data() for index in self.gui.selection_model.selectedRows(0)]
        z_comb = [self.gui.red_channel.currentIndex(),
                  self.gui.green_channel.currentIndex(),
                  self.gui.blue_channel.currentIndex()]
        with (pydetecdiv_project(PyDetecDiv().project_name) as project):
            print('Saving run')
            run = self.save_run(project, 'predict', {'fov': fov_names,
                                                     'network': module.__name__,
                                                     'weights': weights,
                                                     'class_names': self.class_names,
                                                     'red': self.gui.red_channel.currentIndex(),
                                                     'green': self.gui.green_channel.currentIndex(),
                                                     'blue': self.gui.blue_channel.currentIndex()
                                                     })
            for fov_name in fov_names:
                fov = project.get_named_object('FOV', fov_name)
                print(f'Getting image data for FOV = {fov_name}')
                imgdata = fov.image_resource().image_resource_data()
                n_sections = np.max([int(len(fov.roi_list) // batch_size), 1])
                print(f'ROI list in {n_sections} batches')
                for batch in np.array_split(np.array(fov.roi_list), n_sections):
                    if len(input_shape) == 4:
                        x, y = input_shape[1:3]
                        for t in range(imgdata.sizeT):
                            roi_images = get_rgb_images_from_stacks(imgdata, batch, t, z=z_comb)
                            img_array = tf.image.resize(roi_images, (y, x), method='nearest')
                            predictions = model.predict(img_array)
                            # print(predictions.shape)
                            for roi, pred in zip(batch, predictions):
                                Results().save(project, run, roi, t, pred[0, 0, ...], self.class_names)
                    else:
                        x, y = input_shape[2:4]
                        seqlen = self.gui.seq_length.value()
                        for t in range(0, imgdata.sizeT, seqlen):
                            print(f'Sequence from {t} to {t + seqlen - 1}')
                            print('Reading batch')
                            roi_sequences = get_images_sequences(imgdata, batch, t, seqlen=seqlen, z=z_comb)
                            print('Sequence loaded, resizing images')
                            img_array = tf.convert_to_tensor(
                                [tf.image.resize(i, (y, x), method='nearest') for i in roi_sequences])
                            print('Classification')
                            predictions = model.predict(img_array)
                            # print(predictions.shape)
                            print('Saving results')
                            for roi, pred in zip(batch, predictions):
                                for frame, scores in enumerate(pred):
                                    Results().save(project, run, roi, t + frame, scores, self.class_names)
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
            PyDetecDiv().saved_rois.connect(self.set_table_view)
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
                self.gui.classifier_selectionLayout.setRowVisible(1, True)
                self.gui.preprocessing.show()
                self.gui.misc_box.show()
                self.gui.epochs.show()
                self.gui.network.setEditable(False)
                self.gui.classes.setReadOnly(True)
                self.gui.datasets.show()
            case 3:
                # Classify ROIs
                self.gui.roi_selection.show()
                self.gui.roi_sample.hide()
                self.gui.classifier_selectionLayout.setRowVisible(1, True)
                self.gui.preprocessing.show()
                self.gui.misc_box.show()
                self.gui.epochs.hide()
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

        roi_list = prepare_data(get_annotated_rois(), seqlen)
        random.shuffle(roi_list)
        num_training = int(self.gui.training_data.value() * len(roi_list))
        num_validation = int(self.gui.validation_data.value() * len(roi_list))

        module = self.gui.network.currentData()
        print(module.__name__)
        model = module.load_model(load_weights=False)
        print('Loading weights')
        weights = self.gui.weights.currentData()
        if weights:
            module.loadWeights(model, filename=self.gui.weights.currentData())

        print('Compiling model')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        input_shape = model.layers[0].output.shape
        print(model.layers[-1].output.shape)

        if len(input_shape) == 4:
            img_size = (input_shape[1], input_shape[2])
            print('Training dataset')
            training_dataset = ROIDataset(roi_list[:num_training],
                                          image_size=img_size, class_names=self.class_names, batch_size=batch_size)
            print('Validation dataset')
            validation_dataset = ROIDataset(roi_list[num_training:num_training + num_validation],
                                            image_size=img_size, class_names=self.class_names, batch_size=batch_size)
            print('Test dataset')
            test_dataset = ROIDataset(roi_list[num_training + num_validation:], image_size=img_size,
                                      class_names=self.class_names, batch_size=batch_size)
        else:
            img_size = (input_shape[2], input_shape[3])
            print('Training dataset')
            training_dataset = ROIDataset(roi_list[:num_training], image_size=img_size, class_names=self.class_names,
                                          seqlen=seqlen, batch_size=batch_size)
            print('Validation dataset')
            validation_dataset = ROIDataset(roi_list[num_training:num_training + num_validation],
                                            image_size=img_size, class_names=self.class_names, seqlen=seqlen,
                                            batch_size=batch_size)
            print('Test dataset')
            test_dataset = ROIDataset(roi_list[num_training + num_validation:], image_size=img_size,
                                      class_names=self.class_names, seqlen=seqlen, batch_size=batch_size)

        print(input_shape)

        for r in training_dataset.__iter__():
            print(r[0].shape, r[1].shape)

        histories = {'Training': model.fit(training_dataset, epochs=epochs,
                                           # steps_per_epoch=num_training, #//batch_size,
                                           # callbacks=callbacks,
                                           validation_data=validation_dataset,
                                           verbose=2,
                                           # workers=4, use_multiprocessing=True
                                           )}
        print('Not implemented')


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
                                                           C=0, Z=z[0], T=0,
                                                           drift=None)).stretch_contrast(),
                                Image(imgdata.image_memmap(sliceX=slice(roi.x, roi.x + roi.width),
                                                           sliceY=slice(roi.y, roi.y + roi.height),
                                                           C=0, Z=z[1], T=0,
                                                           drift=None)).stretch_contrast(),
                                Image(imgdata.image_memmap(sliceX=slice(roi.x, roi.x + roi.width),
                                                           sliceY=slice(roi.y, roi.y + roi.height),
                                                           C=0, Z=z[2], T=0,
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
