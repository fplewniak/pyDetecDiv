"""
Classes and functions to manage GUI for ROI HDF5 data source creation
"""
from PySide6.QtCore import Signal

from pydetecdiv.utils import check
from pydetecdiv.app import set_connections, PyDetecDiv
from pydetecdiv.app.gui.tools import ToolDialog


class Create_ROI_HDF5Dialog(ToolDialog):
    """
    Dialog window to create ROI HDF5 file
    """
    progress = Signal(int)

    def __init__(self, tool):
        super().__init__(tool, title='Create ROI HDF5 file')

        self.destination = self.addGroupBox(title='HDF5 destination file',
                                            parameters=[
                                                self.tool.parameters.hdf5_file,
                                                self.tool.parameters.time_first,
                                                ],
                                            widget_args={
                                                'hdf5_file': {'filters': ["All files (*)", "HDF5 (*.h5 *.hdf5)", ],
                                                              'selected_filter': 1,
                                                              'min_width': 200},
                                                }
                                            )
        self.other_parameters = self.addGroupBox(title='',
                                                 parameters=[
                                                     self.tool.parameters.seqlen,
                                                     self.tool.parameters.annotations,
                                                     ],
                                                 widget_args={'annotations': {'enable': check.if_annotations,
                                                                              'default': check.if_annotations,}}
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
                                               widget_args={'classification': {'enable': check.if_class_scheme}}
                                               )

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([
            self.destination,
            self.other_parameters,
            self.channels,
            self.classification,
            self.button_box,
            ])

        set_connections({
            self.button_box.accepted: lambda: self.wait_for_command(
                    msg=f'Creating {self.tool.parameters.hdf5_file.value}',
                    cancel_msg=None,
                    ),
            self.button_box.rejected: lambda: print(self.tool.parameters.hdf5_file.value),
            PyDetecDiv.app.project_selected: [lambda: self.tool.parameters.annotations.reset(),
                                              lambda: self.tool.parameters.classification.reset()]
            })
        #
        # self.tool.update_channels()
        # self.tool.update_classification()

        self.fit_to_contents()
        self.exec()

    # def accept(self) -> None:
    #     """
    #     Launch the import and wait for completion
    #     """
    #     wait_dialog = WaitDialog(f'{self.tool.parameters.hdf5_file.value}', self, progress_bar=True, title='Creating ROI HDF5')
    #     wait_dialog.wait_for(self.create_file)
    #     self.close()
    #
    # def create_file(self) -> None:
    #     for i in self.tool.callback():
    #         self.progress.emit(i)
    #     self.finished.emit(True)
