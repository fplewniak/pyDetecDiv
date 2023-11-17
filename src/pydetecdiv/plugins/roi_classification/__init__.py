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
        model = netCNNdiv1.load_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        class_names = ['clog', 'dead', 'empty', 'large', 'small', 'unbud']

        # self.test_model(model)

    # def dummy(self, model, class_names):

        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            for fov_name in [index.data() for index in self.gui.selection_model.selectedRows(0)]:
                fov = project.get_named_object('FOV', fov_name)
                imgdata = fov.image_resource().image_resource_data()

                for t in range(fov.image_resource().sizeT - 4):
                    n=0
                    roi_images = self.get_rgb_images_from_stacks(imgdata, fov.roi_list, t)
                    # roi_images = self.get_images_sequences(imgdata, fov.roi_list, t)

                    img_array = np.array(list(roi_images.values()))
                    # print(img_array.shape, img_array.dtype)
                    img_array = np.array([tf.image.resize(i, (224, 224), method='nearest') for i in img_array])
                    # print(img_array.shape, img_array.dtype)
                    # print(np.max(img_array[n][0]), np.min(img_array[n][0]), np.median(img_array[n][0]), np.mean(img_array[n][0]))
                    print(np.max(img_array[n]), np.min(img_array[n]), np.median(img_array[n]), np.mean(img_array[n]))

                    predictions = model.predict(img_array)

                    for p, roi in zip(predictions, roi_images):
                        # print(f'{roi} {t}: {p[0][0]}')
                        max_score, max_index = max((value, index) for index, value in enumerate(p[0][0]))
                        print(f'{roi} {t}: {class_names[max_index]} ({max_score:.2f})')

                        # for c, s in enumerate(p[0, 0]):
                        #     print(f'          {class_names[c]}: {s:.2f} ',)

                    if PyDetecDiv().main_window.active_subwindow:
                        for i in range(6):
                            ndx = random.randint(0, len(fov.roi_list) - 1)
                            PyDetecDiv().main_window.active_subwindow.show_image(
                                np.array(list(img_array)[ndx]),
                                format_=QImage.Format_RGB888)
                            gray = cv2.cvtColor(np.array(list(img_array)[ndx]), cv2.COLOR_BGR2GRAY)
                            df = pandas.DataFrame()
                            df['L'] = pandas.Series(np.array(gray).flatten())
                            df['r'] = pandas.Series(np.array(list(img_array[ndx]))[...,0].flatten())
                            df['g'] = pandas.Series(np.array(list(img_array[ndx]))[...,1].flatten())
                            df['b'] = pandas.Series(np.array(list(img_array[ndx]))[...,2].flatten())
                            plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow)
                            PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Histogram')
                            df.plot(ax=plot_viewer.axes, kind='hist', bins=32)
                            plot_viewer.canvas.draw()
                            PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)



                    # if PyDetecDiv().main_window.active_subwindow:
                    #     PyDetecDiv().main_window.active_subwindow.show_image(
                    #         tf.image.convert_image_dtype(list(img_array[n])[0], dtype=tf.uint8, saturate=False),
                    #         format_=QImage.Format_RGB888)
                    #
                    #     # df = pandas.DataFrame(np.array(list(img_array.numpy())[n]).flatten())
                    #     df = pandas.DataFrame()
                    #     df['r'] = pandas.Series(np.array(list(img_array[n])[0])[...,0].flatten())
                    #     df['g'] = pandas.Series(np.array(list(img_array[n])[0])[...,1].flatten())
                    #     df['b'] = pandas.Series(np.array(list(img_array[n])[0])[...,2].flatten())
                    #     plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow)
                    #     PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Histogram')
                    #     df.plot(ax=plot_viewer.axes, kind='hist')
                    #     plot_viewer.canvas.draw()
                    #     PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)

                    # if PyDetecDiv().main_window.active_subwindow:
                    #     for d in data:
                    #         PyDetecDiv().main_window.active_subwindow.show_image(
                    #             tf.image.convert_image_dtype(list(d), dtype=tf.uint8, saturate=False),
                    #             format_=QImage.Format_RGB888)

    def normalize_data(self, img):
        # print(img.shape, img.dtype)
        # print(np.max(img), np.min(img), np.median(img), np.mean(img))
        ##############################
        # Adjusting contrast
        # img = tf.convert_to_tensor(np.expand_dims(img, axis=-1))
        # img = tf.image.adjust_contrast(img, 2.0)
        # img = img[..., 0]
        ##############################
        # Contrast stretching
        # img = exposure.rescale_intensity(img, in_range='image', out_range=(0, 1))
        # img = exposure.rescale_intensity(img, in_range='image', out_range='uint8')
        img = np.array(img)
        qlow, qhi = np.quantile(img[img > 0], [0.001, 0.999])
        img = exposure.rescale_intensity(img, in_range=(qlow, qhi))
        ##############################
        # Histogram adaptive equalization
        # img = tf.image.convert_image_dtype(img, dtype=tf.float64, saturate=False)
        # img = exposure.equalize_adapthist(img)
        ##############################
        # Histogram equalization
        # img = tf.image.convert_image_dtype(img, dtype=tf.float64, saturate=False)
        # img = exposure.equalize_hist(img)
        ##############################
        ##############################
        # Sigmoid correction
        # img = exposure.adjust_sigmoid(img)
        ##############################
        return tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=False)


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
        image1 = imgdata.image(T=t, Z=0)
        image2 = imgdata.image(T=t, Z=1)
        image3 = imgdata.image(T=t, Z=2)

        roi_images = {roi.name:
            np.stack((
                self.normalize_data(
                    image1[slice(roi.y, roi.y + roi.height), slice(roi.x, roi.x + roi.width)]),
                self.normalize_data(
                    image2[slice(roi.y, roi.y + roi.height), slice(roi.x, roi.x + roi.width)]),
                self.normalize_data(
                    image3[slice(roi.y, roi.y + roi.height), slice(roi.x, roi.x + roi.width)])),
                axis=-1) for roi in roi_list}

        return roi_images

    def get_images_sequences(self, imgdata, roi_list, t):
        set1 = self.get_rgb_images_from_stacks(imgdata, roi_list, t)
        set2 = self.get_rgb_images_from_stacks(imgdata, roi_list, t+1)
        set3 = self.get_rgb_images_from_stacks(imgdata, roi_list, t+2)
        set4 = self.get_rgb_images_from_stacks(imgdata, roi_list, t+3)
        roi_sequence = {roi.name:
            np.stack((set1[roi.name], set2[roi.name], set3[roi.name], set4[roi.name]),
                axis=0) for roi in roi_list}
        print('roi sequence', np.array(list(roi_sequence.values())).shape)
        return roi_sequence

    def test_model(self, model):
        import matplotlib.pyplot as plt
        from tifffile import tifffile


        model = netCNN_div1_10.load_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        images = np.array(['/NAS/DataGS02/Fred/div_1/trainingdataset/images/small/Pos0_1_221_frame_0410.tif',
                           '/NAS/DataGS02/Fred/div_1/trainingdataset/images/large/Pos0_1_83_frame_0211.tif',
                           '/NAS/DataGS02/Fred/div_1/trainingdataset/images/empty/Pos0_1_47_frame_0018.tif'
                           ])

        class_names = ['clog', 'dead', 'empty', 'large', 'small', 'unbud']

        for i, fichier in enumerate(images):
            data = tifffile.imread(fichier)
            # sequence =  np.stack((data, data, data, data), axis=0)
            # img_array = tf.expand_dims(sequence, 0)  # Create batch axis
            img_array = tf.expand_dims(data, 0)  # Create batch axis
            print(img_array.shape, img_array.dtype)
            img_array = np.array([tf.image.resize(i, (224, 224), method='nearest') for i in img_array])
            print(img_array.shape, img_array.dtype)
            # print(img_array)

            ax = plt.subplot(1, len(images), i + 1)
            plt.imshow(tf.image.convert_image_dtype(img_array[0], dtype=tf.float32, saturate=False))

            print(np.max(img_array[0]), np.min(img_array[0]), np.median(img_array[0]), np.mean(img_array[0]))

            print(img_array.shape, img_array.dtype)
            data, predictions = model.predict(img_array)

            score = predictions[0, 0, 0]

            print(fichier)
            for c, s in enumerate(score):
                print(f'{class_names[c]}: {s:.2f} ', )

        plt.show()

