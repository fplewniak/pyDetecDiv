"""
An example plugin showing how to interact with database
"""
import numpy as np
from PySide6.QtGui import QAction, QImage
from sqlalchemy import Column, Integer, String, ForeignKey
import tensorflow as tf

import pydetecdiv.persistence.sqlalchemy.orm.main
from pydetecdiv import plugins
from pydetecdiv.plugins.roi_classification.gui import ROIselector, ModelSelector
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project

import pydetecdiv.plugins.roi_classification.models.netCNNdiv1 as netCNNdiv1

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

        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            for fov_name in [index.data() for index in self.gui.selection_model.selectedRows(0)]:
                fov = project.get_named_object('FOV', fov_name)
                imgdata = fov.image_resource().image_resource_data()

                for t in range(fov.image_resource().sizeT):
                    image = np.stack((
                        imgdata.image(T=t, Z=0),
                        imgdata.image(T=t, Z=1),
                        imgdata.image(T=t, Z=2)),
                        axis=-1)

                    roi_images = {roi.name: image[slice(roi.y, roi.y + roi.height), slice(roi.x, roi.x + roi.width), :]
                                  for roi in fov.roi_list}

                    img_array = np.array(list(roi_images.values()))
                    img_array = tf.image.resize(img_array, (224,224))
                    print(img_array.shape)
                    predictions = model.predict(img_array)
                    for p, roi in zip(predictions, roi_images):
                        max_score, max_index = max((value, index) for index, value in enumerate(p[0, 0]))
                        print(f'{roi} {t}: {class_names[max_index]} ({max_score})')
                        # for c, s in enumerate(p[0, 0]):
                        #     print(f'          {class_names[c]}: {s:.2f} ',)

                    # if PyDetecDiv().main_window.active_subwindow:
                    #     PyDetecDiv().main_window.active_subwindow.show_image(
                    #         (np.array(list(roi_images.values())[0]) / 255).astype(np.uint8),
                    #         format_=QImage.Format_RGB888)

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
