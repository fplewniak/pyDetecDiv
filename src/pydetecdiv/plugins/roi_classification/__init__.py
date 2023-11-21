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
from pydetecdiv.domain.Image import Image, DType

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
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        class_names = ['clog', 'dead', 'empty', 'large', 'small', 'unbud']

        # self.test_model(model)
        self.test_Image()

    def dummy(self, model, class_names):

        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            for fov_name in [index.data() for index in self.gui.selection_model.selectedRows(0)]:
                fov = project.get_named_object('FOV', fov_name)
                imgdata = fov.image_resource().image_resource_data()
                roi_idx = random.randint(0, len(fov.roi_list) - 1)
                roi_list = [fov.roi_list[roi_idx]]

                # for t in range(fov.image_resource().sizeT - 4):
                for t in range(1):
                    n=0
                    # roi_images = self.get_rgb_images_from_stacks(imgdata, fov.roi_list, t)
                    roi_images = self.get_images_sequences(imgdata, roi_list, t)

                    img_array = np.array(list(roi_images.values()))
                    # print(img_array.shape, img_array.dtype)
                    img_array = np.array([tf.image.resize(i, (224, 224), method='nearest') for i in img_array])
                    # print(img_array.shape, img_array.dtype)
                    # print(np.max(img_array[n][0]), np.min(img_array[n][0]), np.median(img_array[n][0]), np.mean(img_array[n][0]))
                    # print(np.max(img_array[n]), np.min(img_array[n]), np.median(img_array[n]), np.mean(img_array[n]))

                    data, predictions = model.predict(img_array)

                    for p, roi in zip(predictions, roi_images):
                        # print(f'{roi} {t}: {p[0]}')
                        max_score, max_index = max((value, index) for index, value in enumerate(p[0]))
                        print(f'{roi} {t}: {class_names[max_index]} ({max_score:.2f})')

                        # for c, s in enumerate(p[0, 0]):
                        #     print(f'          {class_names[c]}: {s:.2f} ',)

                    if PyDetecDiv().main_window.active_subwindow:
                        if t == 0:
                            for i in range(1):
                                ndx = random.randint(0, len(roi_list) - 1)
                                PyDetecDiv().main_window.active_subwindow.show_image(
                                    np.array(list(img_array)[ndx][0]), title=class_names[max_index],
                                    format_=QImage.Format_RGB888)
                                gray = cv2.cvtColor(np.array(list(img_array)[ndx][0]), cv2.COLOR_BGR2GRAY)
                                df = pandas.DataFrame()
                                df['L'] = pandas.Series(np.array(gray).flatten())
                                df['r'] = pandas.Series(np.array(list(img_array[ndx][0]))[...,0].flatten())
                                df['g'] = pandas.Series(np.array(list(img_array[ndx][0]))[...,1].flatten())
                                df['b'] = pandas.Series(np.array(list(img_array[ndx][0]))[...,2].flatten())
                                plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow)
                                PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Histogram')
                                df['L'].plot(ax=plot_viewer.axes, kind='hist', bins=64, color='black', alpha=1)
                                df['r'].plot(ax=plot_viewer.axes, kind='hist', bins=64, color='red', alpha=0.8)
                                df['g'].plot(ax=plot_viewer.axes, kind='hist', bins=64, color='green', alpha=0.8)
                                df['b'].plot(ax=plot_viewer.axes, kind='hist', bins=64, color='blue', alpha=0.8)
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
        # qlow, qhi = np.quantile(img[img > 0.0], [0.001, 0.999])
        qlow, qhi = np.quantile(img[img > 0], [0.01, 0.99])
        # qlow, qhi = np.min(img), np.max(img)
        # qhi = np.max(img)
        img = exposure.rescale_intensity(img, in_range=(qlow, qhi))
        ##############################
        # # Histogram adaptive equalization
        # img = tf.image.convert_image_dtype(img, dtype=tf.float64, saturate=True)
        # img = exposure.equalize_adapthist(np.array(img))
        ##############################
        # Histogram equalization
        # img = tf.image.convert_image_dtype(img, dtype=tf.float64, saturate=True)
        # img = exposure.equalize_hist(np.array(img))
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
        frame_dicts = [self.get_rgb_images_from_stacks(imgdata, roi_list, t) for f in range(t, min(imgdata.sizeT, 50))]
        # set1 = self.get_rgb_images_from_stacks(imgdata, roi_list, t)
        # set2 = self.get_rgb_images_from_stacks(imgdata, roi_list, t+1)
        # set3 = self.get_rgb_images_from_stacks(imgdata, roi_list, t+2)
        # set4 = self.get_rgb_images_from_stacks(imgdata, roi_list, t+3)
        roi_sequence = {roi.name:
            np.stack([f[roi.name] for f in frame_dicts],
                axis=0) for roi in roi_list}
        print('roi sequence', np.array(list(roi_sequence.values())).shape)
        return roi_sequence

    def test_model(self, model):
        import matplotlib.pyplot as plt
        from tifffile import tifffile

        images = np.array(['/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/small/Pos0_1_221_frame_0410.tif',
                           '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/large/Pos0_1_83_frame_0211.tif',
                           '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/empty/Pos0_1_47_frame_0018.tif'
                           ])

        class_names = ['clog', 'dead', 'empty', 'large', 'small', 'unbud']

        fig, axs = plt.subplots(frameon=False)
        spec = fig.add_gridspec(ncols=2, nrows=len(images))

        for i, fichier in enumerate(images):
            data = tifffile.imread(fichier)
            # sequence =  np.stack((data, data, data, data), axis=0)
            sequence =  np.stack((data,), axis=0)
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
            df['r'] = pandas.Series(np.array(img_array[0][0])[...,0].flatten())
            df['g'] = pandas.Series(np.array(img_array[0][0])[...,1].flatten())
            df['b'] = pandas.Series(np.array(img_array[0][0])[...,2].flatten())
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


    def test_Image(self):
        import matplotlib.pyplot as plt
        from tifffile import tifffile

        images = np.array(
            ['/NAS/Data/BioImageIT/TestTrainingSet/Grayscale/Pos16_empty_channel00_z00_frame_0000.tif',
             '/NAS/Data/BioImageIT/TestTrainingSet/Grayscale/Pos16_empty_channel00_z01_frame_0000.tif',
             '/NAS/Data/BioImageIT/TestTrainingSet/Grayscale/Pos16_empty_channel00_z02_frame_0000.tif',
             '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/small/Pos0_1_221_frame_0410.tif',
             '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/large/Pos0_1_83_frame_0211.tif',
             '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/empty/Pos0_1_47_frame_0018.tif'
             ])

        image1 = Image(tifffile.imread(images[0]))
        image2 = Image(tifffile.imread(images[1]))
        image3 = Image(tifffile.imread(images[2]))

        image_rgb = Image(tifffile.imread(images[3]))

        print(image1.shape)
        print(image1.as_array().dtype)
        print(image1.as_tensor().dtype)

        resized = image1.resize((200, 200), method='nearest')
        print(resized.shape)
        print(resized.dtype)

        comp = Image.compose_channels([image1, image2, image3])
        print(comp.shape)

        plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow)
        PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Histogram RGB')
        image_rgb.channel_histogram(ax=plot_viewer.axes)
        plot_viewer.canvas.draw()
        PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)

        plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow)
        PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Histogram gray scale')
        image1.histogram(ax=plot_viewer.axes)
        plot_viewer.canvas.draw()
        PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)

        plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow)
        PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Histogram gray scale')
        image1.stretch_contrast().histogram(ax=plot_viewer.axes, color='blue')
        image1.equalize_hist().histogram(ax=plot_viewer.axes, color='green')
        image1.equalize_hist(adapt=True).histogram(ax=plot_viewer.axes, color='red')
        image1.sigmoid_correction().histogram(ax=plot_viewer.axes, color='yellow')
