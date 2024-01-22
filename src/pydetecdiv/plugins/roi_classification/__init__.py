"""
An example plugin showing how to interact with database
"""
import importlib
import os.path
import pkgutil
import random
import sys

import numpy as np
from PySide6.QtGui import QAction
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import registry
from sqlalchemy.types import JSON
import tensorflow as tf

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, get_plugins_dir
from pydetecdiv.domain.Image import Image, ImgDType

from .gui import ROIclassification
from . import models
from .gui.annotate import open_annotator_from_selection, open_annotator

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
            r, menu, scene = data
            annotate = menu.addAction('Annotate region class')
            annotate.triggered.connect(lambda _: open_annotator_from_selection(self, r, scene))

    def predict(self):
        """
        Method launching the plugin. This may encapsulate (as it is the case here) the call to a GUI or some domain
        functionalities run directly without any further interface.
        """
        self.create_table()
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
                            roi_images = self.get_rgb_images_from_stacks(imgdata, batch, t)
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
                            roi_sequences = self.get_images_sequences(imgdata, batch, t, seqlen=seqlen)
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
            self.gui.button_box.accepted.connect(self.run)
            self.gui.network.currentIndexChanged.connect(self.update_class_names)
            self.gui.action_menu.currentIndexChanged.connect(self.adapt_gui)
            self.gui.action_menu.setCurrentIndex(3)
        self.gui.setVisible(True)

    def adapt_gui(self):
        """
        Modify the appearance of the GUI according to the selected action
        """
        match (self.gui.action_menu.currentIndex()):
            case 0:
                self.gui.roi_selection.hide()
                self.gui.roi_sample.hide()
                self.gui.classifier_selectionLayout.setRowVisible(1, False)
                self.gui.preprocessing.show()
                self.gui.misc_box.hide()
                self.gui.network.setEditable(True)
                self.gui.classes.setReadOnly(False)
            case 1:
                self.gui.roi_selection.hide()
                self.gui.roi_sample.show()
                self.gui.classifier_selectionLayout.setRowVisible(1, False)
                self.gui.preprocessing.hide()
                self.gui.misc_box.hide()
                self.gui.network.setEditable(False)
                self.gui.classes.setReadOnly(True)
            case 2:
                self.gui.roi_selection.hide()
                self.gui.roi_sample.hide()
                self.gui.classifier_selectionLayout.setRowVisible(1, True)
                self.gui.preprocessing.show()
                self.gui.misc_box.show()
                self.gui.network.setEditable(False)
                self.gui.classes.setReadOnly(True)
            case 3:
                self.gui.roi_selection.show()
                self.gui.roi_sample.hide()
                self.gui.classifier_selectionLayout.setRowVisible(1, True)
                self.gui.preprocessing.show()
                self.gui.misc_box.show()
                self.gui.network.setEditable(False)
                self.gui.classes.setReadOnly(True)
            case _:
                pass
        self.gui.resize(self.gui.form.sizeHint())

    def annotate_rois(self):
        """
        Launch the annotator GUI for ROI annotation
        """
        self.create_table()
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
        Launch training a model: select the network, load weights (optional), define the training and validation sets,
        then run the training.
        """
        print('Not implemented')

    def get_rgb_images_from_stacks(self, imgdata, roi_list, t, z=None):
        """
        Combine 3 z-layers of a grayscale image resource into a RGB image where each of the z-layer is a channel
        :param imgdata: the image data resource
        :param roi_list: the list of ROIs
        :param t: the frame index
        :param z: a list of 3 z-layer indices defining the grayscale layers that must be combined as channels
        :return: a tensor of the combined RGB images
        """
        if z is None:
            z = [self.gui.red_channel.currentIndex(),
                 self.gui.green_channel.currentIndex(),
                 self.gui.blue_channel.currentIndex()]
        image1 = Image(imgdata.image(T=t, Z=z[0]))
        image2 = Image(imgdata.image(T=t, Z=z[1]))
        image3 = Image(imgdata.image(T=t, Z=z[2]))

        print(f'Composing for frame {t}')
        roi_images = [Image.compose_channels([image1.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast(),
                                              image2.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast(),
                                              image3.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast()
                                              ]).as_tensor(ImgDType.float32) for roi in roi_list]
        return roi_images

    def get_images_sequences(self, imgdata, roi_list, t, seqlen=None):
        """
        Get a sequence of seqlen images for each roi
        :param imgdata: the image data resource
        :param roi_list: the list of ROIs
        :param t: the starting time point (index of frame)
        :param seqlen: the number of frames
        :return: a tensor containing the sequences for all ROIs
        """
        maxt = min(imgdata.sizeT, t + seqlen) if seqlen else imgdata.sizeT
        roi_sequences = tf.stack([self.get_rgb_images_from_stacks(imgdata, roi_list, f) for f in range(t, maxt)],
                                 axis=1)
        # print('roi sequence', roi_sequences.shape)
        return roi_sequences
