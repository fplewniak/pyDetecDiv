"""
Classes and functions to manage GUI for ROI HDF5 data source creation
"""
from typing import Any

from PySide6.QtWidgets import QMenu

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.core.widgets import set_connections
from pydetecdiv.app.gui.tools import ToolDialog, ToolAction
from pydetecdiv.persistence.project import project_exists


class Create_ROI_HDF5Dialog(ToolDialog):
    """
    Dialog window to create ROI HDF5 file
    """
    def __init__(self, tool):
        super().__init__(tool, title='Create ROI HDF5 file')

        self.tool.parameters.hdf5_file.current_dir = self.tool.working_dir

        self.destination = self.addGroupBox(title='HDF5 destination file',
                                            parameters=[
                                                self.tool.parameters.hdf5_file,
                                                self.tool.parameters.time_first,
                                                ],
                                            widget_args={
                                                'hdf5_file': {'min_width': 200},
                                                }
                                            )
        self.other_parameters = self.addGroupBox(title='',
                                                 parameters=[
                                                     self.tool.parameters.seqlen,
                                                     self.tool.parameters.annotations,
                                                     ],
                                                 )
        self.channels = self.addGroupBox(title='Channels',
                                                 parameters=[
                                                     self.tool.parameters.red_channel,
                                                     self.tool.parameters.green_channel,
                                                     self.tool.parameters.blue_channel,
                                                     ],
                                                 )

        self.classification = self.addGroupBox(title='Classification schema',
                                               parameters=[self.tool.parameters.classification],
                                               )

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([
            self.destination,
            self.other_parameters,
            self.channels,
            self.classification,
            self.button_box,
            ])

        set_connections({self.button_box.accepted: self.tool.create_file,
                         self.button_box.rejected: lambda: print('Rejected'),
                         })

        self.tool.update_channels()
        self.tool.update_classification()

        self.fit_to_contents()
        self.exec()


class Create_ROI_HDF5Action(ToolAction):
    """
    Action triggering ROI HDF5 file creation.
    """
    def __init__(self, tool_name: str, parent: QMenu = None):
        super().__init__("Create ROI HDF5", tool_name, parent)

    def determine_enabled_status(self, **kwargs: dict[str, Any]):
        self.setEnabled(False)
        if project_exists(PyDetecDiv.project_name):
            with pydetecdiv_project(PyDetecDiv.project_name) as project:
                if project.count_objects('ROI') > 0:
                    self.setEnabled(True)

    def launch(self):
        """
        Run training procedure
        """

        Create_ROI_HDF5Dialog(self.tool)
