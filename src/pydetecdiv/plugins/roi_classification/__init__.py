"""
An example plugin showing how to interact with database
"""
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

from .gui import ROIselector, ModelSelector
from .models import div1, netCNNdiv1, netCNN_div1_10

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
        action_select_model = QAction("Select model", submenu)
        action_select_model.triggered.connect(self.model_selector)
        action_launch = QAction("ROI selection", submenu)
        action_launch.triggered.connect(self.roi_selector)
        submenu.addAction(action_select_model)
        submenu.addAction(action_launch)

    def launch(self):
        """
        Method launching the plugin. This may encapsulate (as it is the case here) the call to a GUI or some domain
        functionalities run directly without any further interface.
        """
        model = div1.load_model()
        # print(model.layers)
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

        # self.test_model(model)
        # self.test_Image()

        # def dummy(self, model, class_names):

        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            for fov_name in [index.data() for index in self.gui.selection_model.selectedRows(0)]:
                fov = project.get_named_object('FOV', fov_name)
                imgdata = fov.image_resource().image_resource_data()
                roi_idx = [random.randint(0, len(fov.roi_list) - 1) for _ in range(5)]
                roi_list = [fov.roi_list[r] for r in roi_idx]
                roi_names = [roi.name for roi in roi_list]

                # for t in range(fov.image_resource().sizeT - 4):
                for t in range(1):
                    n = 0
                    # roi_images = self.get_rgb_images_from_stacks(imgdata, fov.roi_list, t)
                    roi_sequences = self.get_images_sequences(imgdata, roi_list, t)
                    print(np.min(roi_sequences), np.max(roi_sequences))

                    img_array = tf.convert_to_tensor(
                        [tf.image.resize(i, (224, 224), method='nearest') for i in roi_sequences])
                    # img_array = tf.math.multiply(img_array, 255.0)
                    data, predictions = model.predict(img_array)

                    if t == 0:
                        plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow,
                                                    rows=len(roi_list), columns=6)
                        PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'sequences')
                        for i, roi in enumerate(roi_list):
                            plot_viewer.axes[i, 0].set_title(roi.name)
                            for j, arr in enumerate(list(img_array)[i]):
                                arr = tf.math.divide(arr, 255.0)
                                img = Image(arr).equalize_hist(adapt=True)
                                img.show(plot_viewer.axes[i, j])
                        plot_viewer.canvas.draw()
                        PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)

                    for i, (p, roi) in enumerate(zip(predictions, roi_names)):
                        max_score, max_index = max((value, index) for index, value in enumerate(p[0]))
                        print(f'{roi} {i} {t}: {class_names[max_index]} ({max_score:.2f})')

                        if PyDetecDiv().main_window.active_subwindow:
                            if t == 0:
                                plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, columns=2)
                                PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer,
                                                                                 f'{roi} - {class_names[max_index]}')

                                image = Image(list(img_array)[i][0])
                                image.show(plot_viewer.axes[0])
                                plot_viewer.axes[0].set_title(f'{roi} - {class_names[max_index]}')
                                Image(image.as_tensor(ImgDType.uint8)).channel_histogram(plot_viewer.axes[1], bins=64)

                                plot_viewer.canvas.draw()
                                PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)

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
                self.gui.update_list(project.repository.name)

    def model_selector(self):
        if self.model_gui is None:
            self.model_gui = ModelSelector(PyDetecDiv().main_window)
        self.model_gui.setVisible(True)

    def get_rgb_images_from_stacks(self, imgdata, roi_list, t):
        image1 = Image(imgdata.image(T=t, Z=0))
        image2 = Image(imgdata.image(T=t, Z=1))
        image3 = Image(imgdata.image(T=t, Z=2))

        roi_images = [Image.compose_channels([image1.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast(),
                                              image2.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast(),
                                              image3.crop(roi.y, roi.x, roi.height, roi.width).stretch_contrast()
                                              ]).as_tensor(ImgDType.float32) for roi in roi_list]
        return roi_images

    def get_images_sequences(self, imgdata, roi_list, t):
        roi_sequences = tf.stack(
            [self.get_rgb_images_from_stacks(imgdata, roi_list, t) for frame in range(t, min(imgdata.sizeT, t + 6))],
            axis=1)
        print('roi sequence', roi_sequences.shape)
        return roi_sequences

    def test_model(self, model):
        import matplotlib.pyplot as plt
        from tifffile import tifffile

        images = np.array(
            ['/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/small/Pos0_1_221_frame_0410.tif',
             '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/large/Pos0_1_83_frame_0211.tif',
             '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/empty/Pos0_1_47_frame_0018.tif'
             ])

        class_names = ['clog', 'dead', 'empty', 'large', 'small', 'unbud']

        fig, axs = plt.subplots(frameon=False)
        spec = fig.add_gridspec(ncols=2, nrows=len(images))

        for i, fichier in enumerate(images):
            data = tifffile.imread(fichier)
            # sequence =  np.stack((data, data, data, data), axis=0)
            sequence = np.stack((data,), axis=0)
            img_array = tf.expand_dims(sequence, 0)  # Create batch axis
            # img_array = tf.expand_dims(data, 0)  # Create batch axis
            print(img_array.shape, img_array.dtype)
            img_array = np.array([tf.image.resize(i, (224, 224), method='nearest') for i in img_array])
            print(img_array.shape, img_array.dtype)
            # print(img_array)

            ax0 = fig.add_subplot(spec[i, 0])
            plt.imshow(tf.image.convert_image_dtype(img_array[0][0], dtype=tf.float32, saturate=False))

            ax1 = fig.add_subplot(spec[i, 1])
            gray = cv2.cvtColor(np.array(img_array[0][0]), cv2.COLOR_BGR2GRAY)
            df = pandas.DataFrame()
            df['L'] = pandas.Series(np.array(gray).flatten())
            df['r'] = pandas.Series(np.array(img_array[0][0])[..., 0].flatten())
            df['g'] = pandas.Series(np.array(img_array[0][0])[..., 1].flatten())
            df['b'] = pandas.Series(np.array(img_array[0][0])[..., 2].flatten())
            df['L'].plot(ax=ax1, kind='hist', bins=32, color='black', alpha=0.7)
            df['r'].plot(ax=ax1, kind='hist', bins=32, color='red', alpha=0.7)
            df['g'].plot(ax=ax1, kind='hist', bins=32, color='green', alpha=0.7)
            df['b'].plot(ax=ax1, kind='hist', bins=32, color='blue', alpha=0.7)

            print(img_array.shape, img_array.dtype)
            data, predictions = model.predict(img_array)

            score = predictions[0, 0,]

            print(fichier)
            for c, s in enumerate(score):
                print(f'{class_names[c]}: {s:.2f} ', )

        plt.show()
