from typing import Any

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets import set_connections
from pydetecdiv.app.gui.tools import ToolDialog


class Create_ROI_HDF5Dialog(ToolDialog):
    def __init__(self):
        super().__init__(PyDetecDiv.tools['cnrs.plewniak.roihdf5creator'], title='Create ROI HDF5 file')

        self.tool.parameters['destination_dir'].current_dir = self.tool.working_dir

        self.destination = self.addGroupBox(title='Destination',
                                                parameters=[
                                                    self.tool.parameters['destination_dir'],
                                                    ])
        self.button_box = self.addButtonBox()

        self.arrangeWidgets([self.destination, self.button_box])

        set_connections({self.button_box.accepted: lambda: print(self.tool.name, self.tool.parameters['destination_dir'].value),
                         self.button_box.rejected: lambda: print('Rejected'),
                         })

        self.fit_to_contents()
        self.exec()


class Create_ROI_HDF5(QAction):
    def __init__(self, parent: QWidget):
        super().__init__("Create ROI HDF5", parent)
        self.triggered.connect(Create_ROI_HDF5Dialog)
        self.setEnabled(False)
        parent.addAction(self)

    def enable(self, roi_count: int):
        """
        Enable or disable this action whether there are roi data or not.

        :param roi_count: the number of ROIs in project
        """
        if PyDetecDiv.project_name and (roi_count > 0):
            self.setEnabled(True)
        else:
            self.setEnabled(False)
