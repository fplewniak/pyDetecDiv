"""
Import Data dialog window
"""
from pathlib import Path
from typing import Any
from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog
from pydetecdiv.domain import Project

from pydetecdiv import utils

from pydetecdiv.app import set_connections, PyDetecDiv
from pydetecdiv.app.gui.core.widgets import DictListView
from pydetecdiv.app.gui.tools import ToolDialog
from pydetecdiv.app.tools import Tool


class DataImporter:
    """
    Class defining the functions to count and import data files according to the source type
    """
    def __init__(self):
        self.import_func: Callable[[str | Path, Project], Any] | None = None
        self.count_data: Callable[[str | Path], int] | None = None


class DataImportDialog(ToolDialog):
    """
    Dialog window for importing data files
    """

    def __init__(self, tool: Tool, **kwargs):
        super().__init__(tool, title='Import', **kwargs)

        self.data_sources = self.addGroupBox(title='Data source paths',
                                             parameters=[
                                                 tool.parameters.format,
                                                 tool.parameters.paths,
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
            self.button_box.accepted: lambda: self.wait_for_command(
                    msg=f'Importing image data into {PyDetecDiv.project_name}',
                    cancel_msg='Rollback of image import: please wait'
                    ),
            add_path_button.pressed: self.choose_path,
            })

        self.run_after_process([lambda: PyDetecDiv.app.project_selected.emit(PyDetecDiv.project_name)])

        self.fit_to_contents()
        self.exec()

    def choose_path(self) -> None:
        """
        Choose a path and define counting and importing functions according to the format
        """
        path, data_importer = None, DataImporter()
        match self.tool.parameters.format:
            case 'metadata':
                path = self.choose_metadata_file()
                data_importer.import_func = self.tool.import_metadata
                data_importer.count_data = utils.count_metadata
            case 'NDTiff':
                path = self.choose_NDTiff()
                data_importer.import_func = self.tool.import_ndtiff
                data_importer.count_data = utils.count_ndtiff
            case 'Image files':
                path = self.choose_image_files()
                data_importer.import_func = self.tool.import_image_files
                data_importer.count_data = self.tool.count_image_files
            case 'Image directory':
                path = self.choose_image_dir()
                data_importer.import_func = self.tool.import_image_dir
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

    def choose_image_files(self) -> str:
        """
        Choose image file using QFileDialog
        :return: the selected file name
        """
        filters = ["All files (*)", "TIFF (*.tiff *.tif)", ]
        file_name, _ = QFileDialog.getOpenFileName(self, caption='Choose file', dir=self.tool.working_dir, filter=";;".join(filters),
                                                   selectedFilter="TIFF (*.tiff *.tif)")
        return file_name

    def choose_image_dir(self) -> str:
        """
        Choose image directory using QFileDialog
        :return: the selected directory name
        """
        dir_name = QFileDialog.getExistingDirectory(self, caption='Choose directory', dir=self.tool.working_dir)
        return dir_name

    def choose_NDTiff(self) -> str:
        """
        Choose NDTiff dataset using QFileDialog
        :return: the selected directory name
        """
        dir_name = QFileDialog.getExistingDirectory(self, caption='Choose directory', dir=self.tool.working_dir)
        return dir_name


class AnnotatedROIsImportDialog(ToolDialog):
    """
    Dialog window for importing data files
    """
    def __init__(self, tool: Tool, **kwargs):
        super().__init__(tool, title='Import', **kwargs)

        self.file_name = self.addGroupBox(title='Import annotated ROIs',
                                          parameters=[
                                              tool.parameters.roi_annotation_file,
                                              tool.parameters.classification,
                                              ],
                                          widget_args={'roi_annotation_file':
                                                           {'filters': ["All files (*)", "csv (*.csv)", "tsv (*.tsv)",],
                                                            'selected_filter': 1,
                                                            }
                                                       }
                                          )
        self.button_box = self.addButtonBox()

        self.arrangeWidgets([
            self.file_name,
            self.button_box
            ])

        set_connections({
            self.button_box.accepted    : lambda: self.wait_for_command(
                    msg=f'Importing annotated ROIs into {PyDetecDiv.project_name}',
                    cancel_msg='Rollback of annotations import: please wait'),
            })

        self.fit_to_contents()
        self.exec()
