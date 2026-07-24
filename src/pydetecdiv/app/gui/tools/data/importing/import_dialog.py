"""
Import Data dialog window
"""
from pathlib import Path
from typing import Callable, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog

from pydetecdiv.app import set_connections, pydetecdiv_project, PyDetecDiv, WaitDialog
from pydetecdiv.app.gui.core.widgets import DictListView
from pydetecdiv.app.gui.tools import ToolDialog
from pydetecdiv.app.tools import Tool


class DataImporter:
    def __init__(self):
        self.import_func: Callable[[Path], Any] | None = None
        self.count_data: Callable[[Path], int] | None = None


class DataImportDialog(ToolDialog):
    progress = Signal(int)
    finished = Signal(bool)

    def __init__(self, tool: Tool, **kwargs):
        super().__init__(tool, title='Import', **kwargs)

        self.data_sources = self.addGroupBox(title='Data source paths',
                                             parameters=[
                                                 tool.parameters.paths,
                                                 tool.parameters.format,
                                                 ],
                                             widget_args={'paths': {'widget': DictListView, 'multiselection': True}}
                                             )
        self.button_box = self.addButtonBox()
        add_path_button = self.button_box.addButton('Add source', QDialogButtonBox.ButtonRole.ActionRole)

        self.arrangeWidgets([
            self.data_sources,
            self.button_box
            ])

        set_connections({
            # button_box.accepted    : self.import_files,
            self.button_box.accepted    : self.accept,
            add_path_button.pressed: self.choose_path,
            })

        self.fit_to_contents()
        self.exec()

    def choose_path(self):
        #path, import_func = None, None
        path, data_importer = None, DataImporter()
        match self.tool.parameters.format:
            case 'metadata':
                path = self.choose_metadata_file()
                data_importer.import_func = self.tool.import_metadata
                data_importer.count_data = self.tool.count_metadata
            case 'NDTiff':
                path = self.choose_NDTiff()
                data_importer.import_func = self.tool.import_ndtiff
                data_importer.count_data = self.tool.count_ndtiff
            case 'Image files':
                path = self.choose_image_files()
                data_importer.import_func = self.tool.import_image_files
                data_importer.count_data = self.tool.count_image_files
            case 'Image directory':
                path = self.choose_image_dir()
                data_importer.import_func = self.tool.import_image_dir
                data_importer.count_data = self.tool.count_image_dir

        if path and path is not None:
            self.tool.parameters.paths.add_item({path: data_importer})

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

    def accept(self):
        wait_dialog = WaitDialog(f'Importing data into {PyDetecDiv.project_name}', self,
                                 cancel_msg='Rollback of image import: please wait', progress_bar=True, )
        self.finished.connect(wait_dialog.close_window)
        self.progress.connect(wait_dialog.show_progress)
        wait_dialog.wait_for(self.import_files)
        # self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def import_files(self):
        print('Counting data')
        file_count = 0
        for path, data_importer in self.tool.parameters.paths.items:
            file_count += data_importer.count_data(path)
        print(f'Total files: {file_count}')

        if file_count:
            with pydetecdiv_project(PyDetecDiv.project_name) as project:
                count = 0
                for path, data_importer in self.tool.parameters.paths.items:
                    for i in data_importer.import_func(path, project):
                        self.progress.emit(100 * float(count + i) / float(file_count))
                    count += i
                project.commit()
            PyDetecDiv.app.project_selected.emit(PyDetecDiv.project_name)
        self.tool.parameters.paths.clear()
        self.finished.emit(True)
