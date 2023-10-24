from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QFormLayout, QLabel, QComboBox, QFrame, \
    QDockWidget

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project


class ActionDockWindow(QDockWidget):
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

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close | QDialogButtonBox.Ok, self)
        self.button_box.setCenterButtons(True)

        self.formLayout.addWidget(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)
        PyDetecDiv().project_selected.connect(self.set_choice)
        self.set_choice(PyDetecDiv().project_name)

        self.setWidget(self.form)

        PyDetecDiv().main_window.addDockWidget(Qt.LeftDockWidgetArea, self, Qt.Vertical)

    def accept(self):
        print(f'OK: {self.position_choice.currentText()}')
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
