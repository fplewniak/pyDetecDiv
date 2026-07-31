"""
Classes and functions to manage GUI for ROI HDF5 data source creation
"""
from typing import cast

from PySide6.QtCore import Signal

from pydetecdiv.app import set_connections, WaitDialog
from pydetecdiv.app.gui.tools import ToolDialog
from pydetecdiv.domain.tools.data.hdf5 import ROIseqHDF5creator


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

        set_connections({self.button_box.accepted: self.accept,
                         self.button_box.rejected: lambda: print(self.tool.parameters.hdf5_file.value)
                         })
        #
        # self.tool.update_channels()
        # self.tool.update_classification()

        self.fit_to_contents()
        self.exec()

    def accept(self) -> None:
        """
        Launch the import and wait for completion
        """
        wait_dialog = WaitDialog(f'{self.tool.parameters.hdf5_file.value}', self, progress_bar=True, title='Creating ROI HDF5')
        self.finished.connect(wait_dialog.close_window)
        self.progress.connect(wait_dialog.show_progress)
        wait_dialog.wait_for(self.create_file)
        self.close()

    def create_file(self) -> None:
        for i in self.tool.callback():
            self.progress.emit(i)
        self.finished.emit(True)
