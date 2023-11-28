"""
An example plugin showing how to interact with database
"""
import os.path
import random

import cv2
import numpy as np
import pandas
from PySide6.QtGui import QAction, QImage
from sqlalchemy import Column, Integer, String, ForeignKey
import tensorflow as tf
from skimage import exposure

import pydetecdiv.persistence.sqlalchemy.orm.main
from pydetecdiv import plugins
from pydetecdiv.app.gui.Windows import MatplotViewer
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.domain.Image import Image, ImgDType

from .gui import ROIclassification, ROIselector, ModelSelector
# from .models import div1, netCNNdiv1, netCNN_div1_10

Base = pydetecdiv.persistence.sqlalchemy.orm.main.Base


class Results(Base):
    """
    The DAO defining and handling the table to store results
    """
    __tablename__ = 'roi_classification'
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
        self.model_gui = None

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
        submenu = menu.addMenu(self.name)
        action_launch = QAction("Classify ROIs", submenu)
        action_launch.triggered.connect(self.roi_classification)
        action_train_model = QAction("Train new model", submenu)
        action_train_model.triggered.connect(self.model_selector)
        submenu.addAction(action_launch)
        submenu.addAction(action_train_model)

    def launch(self):
        """
        Method launching the plugin. This may encapsulate (as it is the case here) the call to a GUI or some domain
        functionalities run directly without any further interface.
        """
        module = self.gui.network.currentData()
        model = module.load_model(load_weights=False)
        weights = self.gui.weights.currentData()
        if weights:
            module.loadWeights(model, filename=self.gui.weights.currentData())
        # for p, w in model.get_weight_paths().items():
        #     print(p, w.shape, w.dtype, np.min(w), np.max(w))
        # print(model.trainable_variables[0].name, model.trainable_variables[0].dtype)
        # print(np.min(model.trainable_variables[0]), np.max(model.trainable_variables[0]))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        class_names = ['clog', 'dead', 'empty', 'large', 'small', 'unbud']

        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            for fov_name in [index.data() for index in self.gui.selection_model.selectedRows(0)]:
                fov = project.get_named_object('FOV', fov_name)
                imgdata = fov.image_resource().image_resource_data()
                roi_idx = [random.randint(0, len(fov.roi_list) - 1) for _ in range(5)]
                roi_list = [fov.roi_list[r] for r in roi_idx]
                roi_names = [roi.name for roi in roi_list]

                # roi_images = self.get_rgb_images_from_stacks(imgdata, fov.roi_list, t)
                roi_sequences = self.get_images_sequences(imgdata, roi_list, 0)

                img_array = tf.convert_to_tensor(
                    [tf.image.resize(i, (224, 224), method='nearest') for i in roi_sequences])

                data, predictions = model.predict(img_array)
                print('predictions OK')
                step = 20
                plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, rows=len(roi_list),
                                            columns=1+int(len(img_array[0])/step))
                PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Sequences')
                heatmap_plot = MatplotViewer(PyDetecDiv().main_window.active_subwindow, rows=len(roi_list),)
                PyDetecDiv().main_window.active_subwindow.addTab(heatmap_plot, 'Predictions')

                for i, (pred, roi) in enumerate(zip(predictions, roi_names)):
                    plot_viewer.axes[i, 0].set_ylabel(f'{roi}', fontsize='xx-small')
                    for frame, p in enumerate(pred):
                        if frame % step == 0:
                            max_score, max_index = max((value, index) for index, value in enumerate(p))
                            j = int(frame / step)
                            plot_viewer.axes[i,j].set_title(f'{frame} ({class_names[max_index]})', fontsize='xx-small')
                            plot_viewer.axes[i,j].set_xticks([])
                            plot_viewer.axes[i,j].set_yticks([])
                            img = Image(img_array[i][frame])
                            img.show(plot_viewer.axes[i, j])

                    heatmap_plot.axes[i].set_ylabel(f'{roi}', fontsize='xx-small')
                    heatmap_plot.axes[i].imshow(np.moveaxis(pred, 0, -1))
                    heatmap_plot.axes[i].set_yticks(np.arange(len(class_names)), labels=class_names, fontsize='xx-small')
                    heatmap_plot.axes[i].set_aspect('auto')

                plot_viewer.canvas.draw()
                heatmap_plot.canvas.draw()
                PyDetecDiv().main_window.active_subwindow.setCurrentWidget(heatmap_plot)

                # plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, columns=2)
                # PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, f'{roi} - {class_names[max_index]}')
                # image = Image(img_array[i][0])
                # image.show(plot_viewer.axes[0])
                # image.channel_histogram(plot_viewer.axes[1], bins=64)
                # plot_viewer.axes[0].set_title(f'{roi} - {class_names[max_index]}')
                #
                # plot_viewer.canvas.draw()
                # PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)

    def roi_classification(self):
        if self.gui is None:
            self.gui = ROIclassification(PyDetecDiv().main_window)
            self.set_table_view(PyDetecDiv().project_name)
            PyDetecDiv().project_selected.connect(self.set_table_view)
            PyDetecDiv().saved_rois.connect(self.set_table_view)
            self.gui.button_box.accepted.connect(self.launch)
        self.gui.setVisible(True)

    def roi_selector(self):
        if self.gui is None:
            self.gui = ROIselector(PyDetecDiv().main_window)
            self.set_table_view(PyDetecDiv().project_name)
            PyDetecDiv().project_selected.connect(self.set_table_view)
            PyDetecDiv().saved_rois.connect(self.set_table_view)
            self.gui.button_box.accepted.connect(self.launch)
        self.gui.setVisible(True)

    def set_table_view(self, project_name):
        if project_name:
            with pydetecdiv_project(project_name) as project:
                self.gui.update_list(project)

    def model_selector(self):
        if self.model_gui is None:
            self.model_gui = ModelSelector(PyDetecDiv().main_window)
        self.model_gui.setVisible(True)

    def get_rgb_images_from_stacks(self, imgdata, roi_list, t, z=None):
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

    def get_images_sequences(self, imgdata, roi_list, t):
        roi_sequences = tf.stack(
            [self.get_rgb_images_from_stacks(imgdata, roi_list, f) for f in range(t, min(imgdata.sizeT, t + 150))],
            axis=1)
        print('roi sequence', roi_sequences.shape)
        return roi_sequences

    def test_model(self, model):
        import tifffile

        images = np.array(
            [
                '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/small/Pos0_1_221_frame_0410.tif',
                '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/large/Pos0_1_83_frame_0211.tif',
                '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/empty/Pos0_1_47_frame_0018.tif'
            ])

        class_names = ['clog', 'dead', 'empty', 'large', 'small', 'unbud']

        plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, rows=len(images), columns=2)
        PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Predictions')

        for i, fichier in enumerate(images):
            image = Image(tifffile.imread(fichier))
            sequence = tf.stack((image.as_tensor(),), axis=0)
            # sequence = image.as_tensor()
            img_array = tf.expand_dims(sequence, 0)  # Create batch axis
            # img_array = tf.expand_dims(data, 0)  # Create batch axis
            print(img_array.shape, img_array.dtype)
            img_array = tf.convert_to_tensor([tf.image.resize(i, (224, 224), method='nearest') for i in img_array])

            print(img_array.shape, img_array.dtype)
            # print(img_array)

            data, predictions = model.predict(img_array)

            score = predictions[0, 0,]

            plot_viewer.axes[i][0].set_title(os.path.basename(fichier))
            image.show(ax=plot_viewer.axes[i][0])
            image.channel_histogram(ax=plot_viewer.axes[i][1], bins=64)
            max_score, max_index = max((value, index) for index, value in enumerate(score))
            plot_viewer.axes[i][0].text(1, 5, f'{class_names[max_index]}: {max_score:.2f}',
                                        {'fontsize': 8, 'color': 'yellow'})
            plot_viewer.canvas.draw()
            PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)

            print(fichier)
            for c, s in enumerate(score):
                print(f'{class_names[c]}: {s:.2f}', )
