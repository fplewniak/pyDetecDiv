"""
Import Data dialog window
"""
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog

from pydetecdiv.app import set_connections
from pydetecdiv.app.gui.core.widgets import DictListView
from pydetecdiv.app.gui.tools import ToolDialog
from pydetecdiv.app.tools import Tool


class DataImportDialog(ToolDialog):
    def __init__(self, tool: Tool, **kwargs):
        super().__init__(tool, title='Import', **kwargs)

        self.data_sources = self.addGroupBox(title='Data source paths',
                                             parameters=[
                                                 tool.parameters.paths,
                                                 tool.parameters.format,
                                                 ],
                                             widget_args={'paths': {'widget': DictListView, 'multiselection': True}}
                                             )
        button_box = self.addButtonBox()
        add_path_button = button_box.addButton('Add source', QDialogButtonBox.ButtonRole.ActionRole)

        self.arrangeWidgets([
            self.data_sources,
            button_box
            ])

        set_connections({
            button_box.accepted    : self.tool.import_files,
            add_path_button.pressed: self.choose_path,
            })

        self.fit_to_contents()
        self.exec()

    def choose_path(self):
        path, import_func = None, None
        match self.tool.parameters.format:
            case 'metadata':
                path = self.choose_metadata_file()
                import_func = self.tool.import_metadata
            case 'NDTiff':
                path = self.choose_NDTiff()
                import_func = self.tool.import_ndtiff
            case 'Image files':
                path = self.choose_image_files()
                import_func = self.tool.import_image_files
            case 'Image directory':
                path = self.choose_image_dir()
                import_func = self.tool.import_image_dir

        if path and path is not None:
            self.tool.parameters.paths.add_item({path: import_func})


    def choose_metadata_file(self):
        filters = ["All files (*)", "Text (*.txt)", ]
        file_name, _ = QFileDialog.getOpenFileName(self, caption='Choose file', dir=self.tool.working_dir, filter=";;".join(filters),
                                                   selectedFilter="Text (*.txt)")
        return file_name

    def choose_image_files(self):
        filters = ["All files (*)", "TIFF (*.tiff *.tif)", ]
        file_name, _ = QFileDialog.getOpenFileName(self, caption='Choose file', dir=self.tool.working_dir, filter=";;".join(filters),
                                                   selectedFilter="TIFF (*.tiff *.tif)")
        return file_name

    def choose_image_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, caption='Choose directory', dir=self.tool.working_dir)
        return dir_name

    def choose_NDTiff(self):
        dir_name = QFileDialog.getExistingDirectory(self, caption='Choose directory', dir=self.tool.working_dir)
        return dir_name
