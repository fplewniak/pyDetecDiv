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
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import registry
import tensorflow as tf

import pydetecdiv.persistence.sqlalchemy.orm.main
from pydetecdiv import plugins
from pydetecdiv.app.gui.Windows import MatplotViewer
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, get_plugins_dir
from pydetecdiv.domain.Image import Image, ImgDType

from .gui import ROIclassification, ROIselector, ModelSelector
from . import models

Base = registry().generate_base()


class Results(Base):
    """
    The DAO defining and handling the table to store results
    """
    __tablename__ = 'roi_classification'
    __table_args__ = {'extend_existing': True}
    id_ = Column(Integer, primary_key=True, autoincrement='auto')


class Plugin(plugins.Plugin):
    """
    A class extending plugins.Plugin to handle the example plugin
    """
    id_ = 'gmgm.plewniak.roiclassification'
    version = '1.0.0'
    name = 'Deep learning'
    category = 'ROI classification'

    def __init__(self):
        super().__init__()
        self.menu = None
        self.results = {'predictions': [], 'roi_list': None, 'img_array': None, 'classes': None}
        self.class_names = ['clog', 'dead', 'empty', 'large', 'small', 'unbud']

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
        self.menu = menu.addMenu(self.name)
        action_launch = QAction("Classify ROIs", self.menu)
        action_launch.triggered.connect(self.roi_classification)
        self.menu.addAction(action_launch)
        action_train_model = QAction("Train new model", self.menu)
        action_train_model.triggered.connect(self.train_model)
        self.menu.addAction(action_train_model)
        action_save_results = QAction("Save results", self.menu)
        action_save_results.triggered.connect(self.save_results)
        self.menu.addAction(action_save_results)

    def launch(self):
        """
        Method launching the plugin. This may encapsulate (as it is the case here) the call to a GUI or some domain
        functionalities run directly without any further interface.
        """
        module = self.gui.network.currentData()
        print(module.__name__)
        model = module.load_model(load_weights=False)
        weights = self.gui.weights.currentData()
        if weights:
            module.loadWeights(model, filename=self.gui.weights.currentData())

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        input_shape = model.layers[0].output.shape
        batch_size = self.gui.batch_size.value()
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            self.predictions = []
            for fov_name in [index.data() for index in self.gui.selection_model.selectedRows(0)]:
                fov = project.get_named_object('FOV', fov_name)
                imgdata = fov.image_resource().image_resource_data()
                self.roi_list = fov.roi_list
                n_sections = np.max([int(len(self.roi_list) // batch_size), 1])
                for batch in np.array_split(np.array(self.roi_list), n_sections):
                    if len(input_shape) == 4:
                        x, y = input_shape[1:3]
                        for t in range(imgdata.sizeT):
                            roi_images = self.get_rgb_images_from_stacks(imgdata, batch, t)
                            self.img_array = tf.image.resize(roi_images, (y, x), method='nearest')
                            predictions = model.predict(self.img_array)
                            self.predictions.append(predictions)
                        self.predictions = tf.squeeze(np.moveaxis(np.array(self.predictions), 0, 1))
                        print(np.array(self.predictions).shape)
                    else:
                        x, y = input_shape[2:4]
                        roi_sequences = self.get_images_sequences(imgdata, batch, 0)
                        self.img_array = tf.convert_to_tensor(
                            [tf.image.resize(i, (y, x), method='nearest') for i in roi_sequences])
                        predictions = model.predict(self.img_array)
                        self.predictions = predictions
                        print(self.predictions.shape)
                print('predictions OK')

    def save_results(self):
        """
        Save results in database, creating the necessary table if it does not exist. Here, results are simply the name
        and id_ of the selected FOV
        """
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            Base.metadata.create_all(project.repository.engine)
            new_result = Results()
            project.repository.session.add(new_result)
            project.commit()

    @property
    def predictions(self):
        return self.results['predictions']

    @predictions.setter
    def predictions(self, predictions):
        self.results['predictions'] = predictions

    @property
    def roi_list(self):
        return self.results['roi_list']

    @roi_list.setter
    def roi_list(self, roi_list):
        self.results['roi_list'] = roi_list

    @property
    def img_array(self):
        return self.results['img_array']

    @img_array.setter
    def img_array(self, img_array):
        self.results['img_array'] = img_array

    @property
    def classes(self):
        return self.results['classes']

    @classes.setter
    def classes(self, classes):
        self.results['classes'] = classes

    def roi_classification(self):
        """
        Display the ROI classification docked GUI window
        """
        if self.gui is None:
            self.gui = ROIclassification(PyDetecDiv().main_window)
            for _, name, _ in pkgutil.iter_modules(models.__path__):
                self.gui.network.addItem(name, userData=importlib.import_module(f'.models.{name}', package=__package__))
            for finder, name, _ in pkgutil.iter_modules([os.path.join(get_plugins_dir(), 'roi_classification/models')]):
                loader = finder.find_module(name)
                spec = importlib.util.spec_from_file_location(name, loader.path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                self.gui.network.addItem(name, userData=module)
            self.gui.update_model_weights()
            self.set_table_view(PyDetecDiv().project_name)
            PyDetecDiv().project_selected.connect(self.set_table_view)
            PyDetecDiv().saved_rois.connect(self.set_table_view)
            self.gui.button_box.accepted.connect(self.launch)
        self.gui.setVisible(True)

    def set_table_view(self, project_name):
        """
        Set the content of the Table view to display the available ROIs to classify
        :param project_name: the name of the project
        """
        if project_name:
            with pydetecdiv_project(project_name) as project:
                self.gui.update_list(project)

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
            z = [0, 1, 2]
        image1 = Image(imgdata.image(T=t, Z=self.gui.red_channel.currentIndex()))
        image2 = Image(imgdata.image(T=t, Z=self.gui.green_channel.currentIndex()))
        image3 = Image(imgdata.image(T=t, Z=self.gui.blue_channel.currentIndex()))

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
        print('roi sequence', roi_sequences.shape)
        return roi_sequences
