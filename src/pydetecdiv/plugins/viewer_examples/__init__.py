"""
A showcase plugin showing how to interact with TabbedViewer and ImageViewer objects
"""
import pandas
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QAction
from tifffile import tifffile
import tensorflow as tf

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.Windows import MatplotViewer
from pydetecdiv.domain.Image import Image, ImgDType
from .gui import AddPlotDialog



class Plugin(plugins.Plugin):
    """
    A class extending plugins.Plugin to handle the showcase plugin
    """
    id_ = 'gmgm.plewniak.viewer.addons'
    version = '1.0.0'
    name = 'Viewer add-ons'
    category = 'Plugin examples'

    def addActions(self, menu):
        """
        Overrides the addActions method in order to create a submenu with several actions for the same menu
        :param menu: the parent menu
        :type menu: QMenu
        """
        submenu = menu.addMenu(self.name)
        action_launch = QAction("Plot dialog window", submenu)
        action_launch.triggered.connect(self.launch)
        submenu.addAction(action_launch)

        action_plot = QAction("Demo plots", submenu)
        action_plot.triggered.connect(self.add_plot)
        submenu.addAction(action_plot)

        action_change_pen = QAction("change pen", submenu)
        action_change_pen.triggered.connect(self.change_pen)
        submenu.addAction(action_change_pen)

    def launch(self):
        """
        Launch the AddplotDialog interface
        """
        self.gui = AddPlotDialog(PyDetecDiv().main_window)
        self.gui.button_box.accepted.connect(self.add_plot)
        self.gui.button_box.accepted.connect(self.gui.close)
        self.gui.exec()

    def change_pen(self):
        """
        Toggle the pen style (colour and width) for drawing regions in the current subwindow
        """
        active_subwindow = PyDetecDiv().main_window.mdi_area.activeSubWindow()
        if active_subwindow:
            tab = [tab for tab in PyDetecDiv().main_window.tabs.values() if tab.window == active_subwindow][0]
            if tab.viewer.scene.pen.width() == 2:
                tab.viewer.scene.pen = QPen(Qt.GlobalColor.blue, 6)
            else:
                tab.viewer.scene.pen = QPen(Qt.GlobalColor.cyan, 2)

    def add_plot(self):
        """
        Add a new tab with a dummy plot to the currently active subwindow
        """
        active_subwindow = PyDetecDiv().main_window.mdi_area.activeSubWindow()
        if active_subwindow:
            tab = [tab for tab in PyDetecDiv().main_window.tabs.values() if tab.window == active_subwindow][0]
            x = np.linspace(0, 10, 500)
            y = np.sin(x)
            df = pandas.DataFrame(y)
            tab.show_plot(df, 'Plugin plot')

            images = np.array(
                ['/NAS/Data/BioImageIT/TestTrainingSet/Grayscale/Pos16_empty_channel00_z00_frame_0000.tif',
                 '/NAS/Data/BioImageIT/TestTrainingSet/Grayscale/Pos16_empty_channel00_z01_frame_0000.tif',
                 '/NAS/Data/BioImageIT/TestTrainingSet/Grayscale/Pos16_empty_channel00_z02_frame_0000.tif',
                 '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/small/Pos0_1_221_frame_0410.tif',
                 '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/large/Pos0_1_83_frame_0211.tif',
                 '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/empty/Pos0_1_47_frame_0018.tif',
                 '/NAS/Data/BioImageIT/TestChannels/img_channel000_position001_time000000284_z001.tif',
                 '/NAS/Data/BioImageIT/TestChannels/img_channel001_position001_time000000284_z001.tif',
                 '/NAS/Data/BioImageIT/TestChannels/img_channel002_position001_time000000284_z001.tif',
                 ])

            image1 = Image(tifffile.imread(images[0]))
            image2 = Image(tifffile.imread(images[1]))
            image3 = Image(tifffile.imread(images[2]))

            image_rgb = Image(tifffile.imread(images[3]))

            bright_field = Image(tifffile.imread(images[6]))
            red = Image(tifffile.imread(images[7]))
            green = Image(tifffile.imread(images[8]))
            blue = bright_field
            zeros = Image(tf.zeros_like(blue.tensor))

            image_fluo = Image.compose_channels([Image.mean([red, bright_field]),
                                                 Image.mean([green, bright_field]),
                                                 Image.mean([zeros, bright_field])]
                                                ).equalize_hist(adapt=True)
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

            plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow)
            PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Histogram gray scale')
            image1.histogram(ax=plot_viewer.axes)
            plot_viewer.canvas.draw()
            PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)

            plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, columns=2)
            PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Histogram image correction')
            image1.stretch_contrast().histogram(ax=plot_viewer.axes[0], color='blue')
            image1.sigmoid_correction().histogram(ax=plot_viewer.axes[0], color='yellow')
            plot_viewer.axes[0].legend(['Strech contrast', 'Sigmoid'])
            image1.equalize_hist().histogram(ax=plot_viewer.axes[1], color='green')
            image1.equalize_hist(adapt=True).histogram(ax=plot_viewer.axes[1], color='red')
            plot_viewer.axes[1].set_title('Equalize histogram')
            plot_viewer.axes[1].legend(['plain', 'adaptative'])


            plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, columns=4)
            PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Image & channels')
            plot_viewer.axes[0].imshow(image_rgb.as_array(ImgDType.uint8))
            channels = image_rgb.decompose_channels()
            colours = ['R', 'G', 'B']
            for i, c in enumerate(channels):
                plot_viewer.axes[i + 1].imshow(c.as_array(ImgDType.uint8), cmap='gray')
                plot_viewer.axes[i + 1].set_title(colours[i])

            plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, )
            PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Fluorescence')
            plot_viewer.axes.imshow(image_fluo.as_array(ImgDType.uint8))

            plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, rows=3, columns=2)
            PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Fluorescence histograms')
            red.histogram(ax=plot_viewer.axes[0, 0], color='red', bins=64)
            green.histogram(ax=plot_viewer.axes[1, 0], color='green', bins=64)
            blue.histogram(ax=plot_viewer.axes[2, 0], color='blue', bins=64)

            colours = ['red', 'green', 'blue']
            for i, c in enumerate(image_fluo.decompose_channels()):
                c.histogram(ax=plot_viewer.axes[i, 1], color=colours[i], bins=64)

            PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)
