"""
A DockWidget class extension to handle the interface to the Example plugin's 'create and save results' action.
"""
from PySide6.QtCore import Qt, QStringListModel
from PySide6.QtWidgets import QDialogButtonBox, QFormLayout, QLabel, QComboBox, QFrame, QDockWidget, QListView
import sqlalchemy

import pydetecdiv
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.utils import singleton


@singleton
class ActionDockWindow(QDockWidget):
    """
    A DockWidget to host the GUI for Example plugin's for Action1 (create and save results)
    This is a singleton to avoid creating more than one window, but this is not compulsory and there may be several
    instance of such a window for a single plugin if needed.
    """

    def __init__(self):
        super().__init__(PyDetecDiv().main_window)
        self.setWindowTitle('Example plugin')
        self.setObjectName('Example plugin')

        self.form = QFrame()
        self.form.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.formLayout = QFormLayout(self.form)

        self.position_label = QLabel('Position', self.form)
        self.position_choice = QComboBox(self.form)
        self.formLayout.addRow(self.position_label, self.position_choice)

        self.list_view = QListView(self.form)
        self.list_model = QStringListModel()
        self.list_view.setModel(self.list_model)

        self.formLayout.addRow(self.list_view)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close | QDialogButtonBox.Ok, self)
        self.button_box.setCenterButtons(True)

        self.formLayout.addRow(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)
        PyDetecDiv().project_selected.connect(self.set_choice)
        PyDetecDiv().project_selected.connect(self.get_saved_results)
        self.set_choice(PyDetecDiv().project_name)

        self.setWidget(self.form)

        PyDetecDiv().main_window.addDockWidget(Qt.LeftDockWidgetArea, self, Qt.Vertical)

    def accept(self):
        """
        Change the window title when a new position has been chosen
        """
        self.setWindowTitle(f'Example plugin/{self.position_choice.currentText()}')

    def set_choice(self, p_name):
        """
        Set the available values for FOVs, datasets and channels given a project name

        :param p_name: the project name
        :type p_name: str
        """
        with pydetecdiv_project(p_name) as project:
            self.position_choice.clear()
            if project.count_objects('FOV'):
                self.position_choice.addItems(sorted([fov.name for fov in project.get_objects('FOV')]))

    def get_saved_results(self):
        if PyDetecDiv().project_name:
            with pydetecdiv_project(PyDetecDiv().project_name) as project:
                if sqlalchemy.inspect(project.repository.engine).has_table(pydetecdiv.plugins.example.Results.__tablename__):
                    self.list_model.setStringList([': '.join([str(r.id_), project.get_object('FOV', r.fov).name])
                                                   for r in project.repository.session.query(pydetecdiv.plugins.example.Results).all()])
        else:
            self.list_model.setStringList([])
