from PySide6.QtWidgets import QDialogButtonBox, QFileDialog

from pydetecdiv.app.tools import Tool

from pydetecdiv import utils

from pydetecdiv.app.gui.core.widgets import DictListView, set_connections
from pydetecdiv.app.gui.tools import ToolDialog
from pydetecdiv.app.gui.tools.data.importing.import_dialog import DataImporter


class Convert2NDTiffDialog(ToolDialog):
    """
    Dialog window to convert TIFF image files to NDTiff format
    """

    def __init__(self, tool: Tool):
        super().__init__(tool, title=tool.title)

        self.data_sources = self.addGroupBox(title='Data source paths',
                                             parameters=[
                                                 tool.parameters.format,
                                                 tool.parameters.paths,
                                                 ],
                                             widget_args={'paths': {'widget': DictListView, 'multiselection': True}}
                                             )
        self.destination = self.addGroupBox(title='Destination path',
                                            parameters=[
                                                tool.parameters.destination,
                                                ])

        self.button_box = self.addButtonBox()
        add_path_button = self.button_box.addButton('Add source', QDialogButtonBox.ButtonRole.ActionRole)

        self.arrangeWidgets([
            self.data_sources,
            self.destination,
            self.button_box,
            ])

        set_connections({
            self.button_box.accepted: lambda: self.wait_for_command(
                    msg=f'Converting image files to NDTiff',
                    cancel_msg='Rollback NDTiff conversion: please wait',
                    ),
            add_path_button.pressed     : self.choose_path,
            })

        self.fit_to_contents()
        self.exec()

    def choose_path(self) -> None:
        """
        Choose a path and define counting and importing functions according to the format
        """
        #path, import_func = None, None
        path, data_importer = None, DataImporter()
        match self.tool.parameters.format:
            case 'metadata':
                path = self.choose_metadata_file()
                data_importer.import_func = self.tool.read_metadata
                data_importer.count_data = utils.count_metadata
            case 'Image directory':
                path = self.choose_image_dir()
                data_importer.import_func = self.tool.read_image_dir
                data_importer.count_data = utils.count_image_dir

        if path and path is not None:
            self.tool.parameters.paths.add_item({path: data_importer})

    def choose_metadata_file(self) -> str:
        """
        Choose a metadata file
        """
        filters = ["All files (*)", "Text (*.txt)", ]
        file_name, _ = QFileDialog.getOpenFileName(self, caption='Choose file', dir=self.tool.working_dir, filter=";;".join(filters),
                                                   selectedFilter="Text (*.txt)")
        return file_name

    def choose_image_dir(self) -> str:
        """
        Choose image directory using QFileDialog
        :return: the selected directory name
        """
        dir_name = QFileDialog.getExistingDirectory(self, caption='Choose directory', dir=self.tool.working_dir)
        return dir_name
