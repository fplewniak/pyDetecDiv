"""
An example plugin showing how to interact with database
"""
from sqlalchemy import Column, Integer, String, ForeignKey

import pydetecdiv.persistence.sqlalchemy.orm.main
from pydetecdiv import plugins
from pydetecdiv.plugins.example.ActionDockWindow import ActionDockWindow
from pydetecdiv.plugins.example.Actions import Action1, Action2
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project

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
    name = 'Example'
    category = 'Plugin examples'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gui = None

    def update_database(self):
        """
        Create the table to save results if it does not exist yet
        """
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            Base.metadata.create_all(project.repository.engine)

    def addActions(self, menu):
        """
        Add actions to Example submenu
        :param menu: the submenu
        :type menu: QMenu
        """
        Action1(menu).triggered.connect(self.show_gui)
        Action2(menu).triggered.connect(self.run)

    def run(self):
        """
        Dummy method showing how to run code in the main plugin file
        """
        print(f'Running plugin action from main plugin code (project: {PyDetecDiv().project_name})')

    def show_gui(self):
        """
        Show the docked window containing the GUI for the example plugin, creating it if it does not exist.
        """
        self.gui = ActionDockWindow()
        self.gui.button_box.accepted.connect(self.save_result)
        # self.gui.get_saved_results()
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
        self.gui.get_saved_results()
