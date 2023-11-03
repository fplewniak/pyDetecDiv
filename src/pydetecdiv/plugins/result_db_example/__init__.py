"""
An example plugin showing how to interact with database
"""
import sqlalchemy
from sqlalchemy import Column, Integer, String, ForeignKey

import pydetecdiv.persistence.sqlalchemy.orm.main
from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from .gui import DockWindow

Base = pydetecdiv.persistence.sqlalchemy.orm.main.Base


class Results(Base):
    """
    The DAO defining and handling the table to store results
    """
    __tablename__ = 'example_results'
    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    name = Column(String)
    fov = Column(Integer, ForeignKey('FOV.id_'), nullable=True, index=True)


class Plugin(plugins.Plugin):
    """
    A class extending plugins.Plugin to handle the example plugin
    """
    id_ = 'gmgm.plewniak.example'
    version = '1.0.0'
    name = 'Results in DB example'
    category = 'Plugin examples'

    def create_table(self):
        """
        Create the table to save results if it does not exist yet
        """
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            Base.metadata.create_all(project.repository.engine)

    def launch(self):
        """
        Method launching the plugin. This may encapsulate (as it is the case here) the call to a GUI or some domain
        functionalities run directly without any further interface.
        """
        self.show_gui()

    def show_gui(self):
        """
        Show the docked window containing the GUI for the example plugin, creating it if it does not exist.
        """
        if self.gui is None:
            self.gui = DockWindow(PyDetecDiv().main_window)
            self.gui.button_box.accepted.connect(self.save_result)
            PyDetecDiv().project_selected.connect(self.show_saved_results)
            PyDetecDiv().saved_rois.connect(self.show_saved_results)
            self.set_choice(PyDetecDiv().project_name)
            PyDetecDiv().project_selected.connect(self.set_choice)
        self.gui.setVisible(True)

    def save_result(self):
        """
        Save results in database, creating the necessary table if it does not exist. Here, results are simply the name
        and id_ of the selected FOV
        """
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            Base.metadata.create_all(project.repository.engine)
            fov = project.get_named_object("FOV", self.gui.position_choice.currentText())
            new_result = Results(name=fov.name, fov=fov.id_)
            project.repository.session.add(new_result)
            project.commit()
        self.show_saved_results()

    def show_saved_results(self, project_name):
        """
        Shows the list of results in the ListView of the GUI (updates the list model)
        """
        if project_name:
            with pydetecdiv_project(project_name) as project:
                if sqlalchemy.inspect(project.repository.engine).has_table(Results.__tablename__):
                    results = [(r.id_, project.get_object('FOV', r.fov)) for r in
                                   project.repository.session.query(Results).all()]
                    string_list = [f'{r}: {fov.name} ({len(fov.roi_list)} ROIs)' for r, fov in results]
                    self.gui.list_model.setStringList(string_list)
        else:
            self.gui.list_model.setStringList([])

    def set_choice(self, p_name):
        """
        Set the available values for FOVs, datasets and channels given a project name

        :param p_name: the project name
        :type p_name: str
        """
        with pydetecdiv_project(p_name) as project:
            self.gui.position_choice.clear()
            if project.count_objects('FOV'):
                self.gui.position_choice.addItems(sorted([fov.name for fov in project.get_objects('FOV')]))
