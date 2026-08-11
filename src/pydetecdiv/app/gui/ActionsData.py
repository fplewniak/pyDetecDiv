"""
Handling actions to open, create and interact with projects
"""
import glob
import json
import os
from subprocess import Popen
from typing import Any, cast

import numpy as np
import polars
import tifffile
from PySide6.QtCore import (QRegularExpression, Signal, QDir, QThread)
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QWidget, QDialogButtonBox, QHBoxLayout, QLineEdit, QPushButton, QFileDialog)
from ndtiff import NDTiffDataset

from pydetecdiv.app import PyDetecDiv, WaitDialog, pydetecdiv_project, MessageDialog
from pydetecdiv.domain.Classification import Classification
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.domain.Project import Project
from pydetecdiv.app.parameters import ChoiceParameter
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.RoiAnnotations import RoiAnnotations
from pydetecdiv.domain.Run import Run

from pydetecdiv.settings import get_config_value
from pydetecdiv import delete_files
from pydetecdiv.app.gui.RawData2FOV import RawData2FOV
import pydetecdiv.app.gui.core.widgets as gui
from pydetecdiv.app.gui.core.widgets.files import FileListChooserDialog


# class ImportMetaDataDialog(FileListChooserDialog):
#     """
#     A dialog window to choose sources for metadata files to import images and create Image resources
#     """
#     progress = Signal(int)
#     chosen_directory = Signal(str)
#     finished = Signal(bool)
#
#     def __init__(self):
#         super().__init__(title='Import image data from metadata', filters=["TXT (*.txt)",
#                                                                            "JSON (*.json)"],
#                          extensions=['*.txt', '*.json'])
#         self.setObjectName('ImportMetaData')
#
#     def accept(self) -> None:
#         """
#         Launches import of data in response to Ok button
#         """
#         wait_dialog = WaitDialog(f'Importing data into {PyDetecDiv.project_name}', self,
#                                  cancel_msg='Rollback of image import: please wait', progress_bar=True, )
#         self.finished.connect(wait_dialog.close_window)
#         self.progress.connect(wait_dialog.show_progress)
#         wait_dialog.wait_for(self.import_data)
#         self.list_model.removeRows(0, self.list_model.rowCount())
#         self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
#
#     def import_data(self) -> None:
#         """
#         Import files whose list is defined by the sources in self.list_model
#         """
#         # destination = os.path.join(self.project_path, 'data', self.destination_directory.currentText())
#         self.progress.emit(0)
#         i = 0.0
#         metadata_file_names = [f for source_path in self.list_model.stringList() for f in glob.glob(source_path) if
#                                os.path.isfile(f)]
#         with pydetecdiv_project(PyDetecDiv.project_name) as project:
#             for metadata_file_name in metadata_file_names:
#                 i += 1.0
#                 project.import_images_from_metadata(metadata_file_name)
#                 self.progress.emit(100 * i / len(metadata_file_names))
#         project.commit()
#         self.finished.emit(True)
#         PyDetecDiv.app.project_selected.emit(PyDetecDiv.project_name)
#
#
# class ImportDataDialog(FileListChooserDialog):
#     """
#     A dialog window to choose sources for image data files to import into the project raw dataset
#     """
#     progress = Signal(int)
#     chosen_directory = Signal(str)
#     finished = Signal(bool)
#
#     def __init__(self):
#         super().__init__(title='Import image data', filters=["TIFF (*.tif *.tiff)",
#                                                              "JPEG (*.jpg *.jpeg)",
#                                                              "PNG (*.png)",
#                                                              "Image files (*.tif *.tiff, *.jpg *.jpeg, *.png)"],
#                          extensions=['*.tiff', '*.tif', '*.jpg', '*.jpeg', '*.png', '*'])
#         self.setObjectName('ImportData')
#         self.project_path = os.path.join(get_config_value('project', 'workspace'), PyDetecDiv.project_name)
#
#     def get_destinations(self) -> list[str]:
#         """
#         Get the list of subdirectories in the destination raw dataset directory
#
#         :return: list of subdirectories in the destination raw dataset directory
#         """
#         return [''] + [d.name for d in os.scandir(os.path.join(self.project_path, 'data')) if d.is_dir()]
#
#     @property
#     def file_list(self) -> list[str]:
#         """
#         Expands all source specification to return a list of files to import
#
#         :return: file name list
#         """
#         file_list = []
#         for source_path in self.list_model.stringList():
#             if source_path.endswith('.txt'):
#                 for metadata_file_name in [f for f in glob.glob(source_path) if os.path.isfile(f)]:
#                     with open(metadata_file_name) as metadata_file:
#                         metadata = json.load(metadata_file)
#                         file_list += [v["FileName"] for k, v in metadata.items() if k.startswith('Metadata-')]
#             else:
#                 file_list += [f for f in glob.glob(source_path) if os.path.isfile(f)]
#         return file_list
#
#     def accept(self) -> None:
#         """
#         Import files whose list is defined by the sources in self.list_model
#         """
#         wait_dialog = WaitDialog(f'Importing data into {PyDetecDiv.project_name}', self,
#                                  cancel_msg='Rollback of image import: please wait', progress_bar=True, )
#         self.finished.connect(wait_dialog.close_window)
#         self.progress.connect(wait_dialog.show_progress)
#         wait_dialog.wait_for(self.import_data)
#         self.list_model.removeRows(0, self.list_model.rowCount())
#         self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
#
#     def import_data(self) -> None:
#         """
#         Import image data from source specified in list_model into project raw dataset and triggers a progress signal
#         with the number of files that have been copied so far
#         """
#         self.progress.emit(0)
#         file_list = self.file_list
#         if len(file_list) == 0:
#             self.finished.emit(True)
#             MessageDialog('No data file to import in specified directories')
#         else:
#             with pydetecdiv_project(PyDetecDiv.project_name) as project:
#                 n_dso0 = project.count_objects('Data')
#                 self.progress.emit(0)
#                 _ = project.import_images(file_list, in_place=True, destination=None)
#                 n_files = 0
#                 n_dso = project.count_objects('Data') - n_dso0
#                 self.progress.emit(int(100 * (n_files + n_dso) / len(file_list)))
#                 if QThread.currentThread().isInterruptionRequested():
#                     project.cancel()
#                 n_raw_data_files = project.count_objects('Data')
#             self.finished.emit(True)
#             PyDetecDiv.app.raw_data_counted.emit(n_raw_data_files)
#
#     def import_data_with_copy(self) -> None:
#         """
#         Import image data from source specified in list_model into project raw dataset and triggers a progress signal
#         with the number of files that have been copied so far
#         """
#         self.progress.emit(0)
#         in_place = self.keep_copy_buttons.button(2).isChecked()
#         destination = os.path.join(self.project_path, 'data', self.destination_directory.currentText())
#         QDir().mkpath(str(destination))
#         file_list = self.file_list
#         if len(file_list) == 0:
#             self.finished.emit(True)
#             MessageDialog('No data file to import in specified directories')
#         else:
#             n_files_to_copy = 0 if in_place else len(file_list)
#             with pydetecdiv_project(PyDetecDiv.project_name) as project:
#                 initial_files = {d.url for d in project.get_objects('Data')}
#                 n_files0 = sum(1 for item in os.listdir(destination) if os.path.isfile(os.path.join(destination, item)))
#                 n_dso0 = project.count_objects('Data')
#                 self.progress.emit(0)
#                 processes = []
#                 for batch in np.array_split(file_list,
#                                             int(len(file_list) / int(get_config_value('project', 'batch'))) + 1):
#                     if len(batch):
#                         process = project.import_images(batch, in_place=in_place,
#                                                         destination=self.destination_directory.currentText())
#                         processes.append(process)
#                         n_files = 0 if in_place else self.count_imported_files(destination, n_files0)
#                         n_dso = project.count_objects('Data') - n_dso0
#                         self.progress.emit(int(100 * (n_files + n_dso) / (len(file_list) + n_files_to_copy)))
#                     if QThread.currentThread().isInterruptionRequested():
#                         self.cancel_import(initial_files, n_files0, project, processes)
#                         return
#                 while (n_files + n_dso) < (len(file_list) + n_files_to_copy):
#                     if QThread.currentThread().isInterruptionRequested():
#                         self.cancel_import(initial_files, n_files0, project, processes)
#                         return
#                     n_files = 0 if in_place else self.count_imported_files(destination, n_files0)
#                     n_dso = project.count_objects('Data') - n_dso0
#                     self.progress.emit(int(100 * (n_files + n_dso) / (len(file_list) + n_files_to_copy)))
#                 n_raw_data_files = project.count_objects('Data')
#             self.finished.emit(True)
#             PyDetecDiv.app.raw_data_counted.emit(n_raw_data_files)
#
#     def count_imported_files(self, destination: str, n_start: int) -> int:
#         """
#         Count imported files in destination directory to assess progress
#
#         :param destination: destination directory which files are imported into
#         :param n_start: the number of files already in the destination directory before import
#         :return: the number of imported files
#         """
#         return sum(1 for item in os.listdir(destination) if os.path.isfile(os.path.join(destination, item))) - n_start
#
#     def cancel_import(self, initial_files: set[str], n_files0: int, project: Project, processes: list[Popen]) -> None:
#         """
#         Manage cancellation of import. Terminate all copy processes before launching deletion of files that were already
#         copied. Then cancel persistence operations on Data objects, and eventually stop the host thread.
#
#         :param initial_files: the set of files before import was started
#         :param n_files0: the initial number of files
#         :param project: the current Project
#         :param processes: import processes that are running and should be cancelled
#         """
#         self.progress.emit(0)
#         in_place = self.keep_copy_buttons.button(2).isChecked()
#         destination = os.path.join(self.project_path, 'data', self.destination_directory.currentText())
#         n_max = self.count_imported_files(destination, n_files0)
#         n_files = 0 if in_place else self.count_imported_files(destination, n_files0)
#         if self.keep_copy_buttons.button(1).isChecked():
#             for process in processes:
#                 process.terminate()
#             while n_files != self.count_imported_files(destination, n_files0):
#                 n_files = self.count_imported_files(destination, n_files0)
#             all_files = {os.path.join(destination, item) for item in os.listdir(destination) if
#                          os.path.isfile(os.path.join(destination, item))}
#             diff = list(all_files.difference(initial_files))
#             imported_batches = np.array_split(diff, int(len(diff) / int(get_config_value('project', 'batch'))) + 1)
#             for cancelled in imported_batches:
#                 delete_files(cancelled)
#                 n_files = 0 if in_place else self.count_imported_files(destination, n_files0)
#                 self.progress.emit(100 - int(100 * n_files / n_max))
#         project.cancel()
#         while n_files > 0:
#             n_files = 0 if in_place else self.count_imported_files(destination, n_files0)
#             self.progress.emit(100 - int(100 * n_files / n_max))
#         self.finished.emit(True)
#
#     def source_list_is_not_empty(self) -> None:
#         """
#         Checks the source list is not empty, enables OK button if not
#         """
#         if self.list_model.rowCount():
#             self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
#         else:
#             self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
#
#     @staticmethod
#     def sub_directory_name_validator() -> QRegularExpressionValidator:
#         """
#         Name validator to filter invalid character in directory name
#
#         :return: the validator
#         """
#         name_filter = QRegularExpression()
#         name_filter.setPattern('\\w[\\w-]*')
#         validator = QRegularExpressionValidator()
#         validator.setRegularExpression(name_filter)
#         return validator
#
#
# class ImportData(QAction):
#     """
#     Action to import raw data images into a project
#     """
#
#     def __init__(self, parent: QWidget):
#         super().__init__(QIcon(":icons/import_images"), "&Import image files", parent)
#         self.triggered.connect(ImportDataDialog)
#         self.setEnabled(False)
#         parent.addAction(self)
#
#
# class ImportMetaData(QAction):
#     """
#     Action to import raw data images into a project
#     """
#
#     def __init__(self, parent: QWidget):
#         super().__init__(QIcon(":icons/import_images"), "&Import metadata files", parent)
#         self.triggered.connect(ImportMetaDataDialog)
#         self.setEnabled(False)
#         parent.addAction(self)
#
#
# class CreateFOV(QAction):
#     """
#     Action to import raw data images into a project
#     """
#
#     def __init__(self, parent: QWidget):
#         super().__init__(QIcon(":icons/import_images"), "Build &Image resources from raw data", parent)
#         self.triggered.connect(RawData2FOV)
#         self.setEnabled(False)
#         parent.addAction(self)
#
#     def enable(self, raw_data_count: int):
#         """
#         Enable or disable this action in the Data menu whether there are raw data or not.
#
#         :param raw_data_count: the number of files in raw dataset
#         """
#         if PyDetecDiv.project_name and (raw_data_count > 0):
#             self.setEnabled(True)
#         else:
#             self.setEnabled(False)
#
#
# class ComputeDriftDialog(gui.Dialog):
#     """
#     A dialog window to run drift correction computations for a selection of FOVs
#     """
#     # progress = Signal(int)
#     finished = Signal(bool)
#
#     def __init__(self, title: str | None = None):
#         super().__init__(title=title)
#
#         self.wait = None
#         self.drift = {}
#
#         self.select_FOV = self.addGroupBox('Select FOVs',
#                                            parameters=[ChoiceParameter(name='FOVs', label='FOV',
#                                                                        items=self.update_fov_list(PyDetecDiv.project_name))],
#                                            widget_args={'FOVs': {'widget': gui.DictListView, 'multiselection': True}}
#                                            )
#
#         self.method_box = self.addGroupBox('Method',
#                                            parameters=[ChoiceParameter(name='Method', label='Method', default='vidstab',
#                                                                        items={'vidstab': None, 'phase correlation': None})],
#                                            )
#
#         self.button_box = self.addButtonBox()
#
#         self.arrangeWidgets([
#             self.select_FOV,
#             self.method_box,
#             self.button_box
#             ])
#
#         gui.set_connections({self.button_box.accepted       : self.accept,
#                              self.button_box.rejected       : self.close,
#                              PyDetecDiv.app.project_selected: self.update_fov_list,
#                              })
#
#         self.exec()
#
#     def update_fov_list(self, project_name: str) -> dict[str, FOV]:
#         """
#         Return the list of FOVs in the project as a dictionary mapping actual FOV objects to their names
#
#         :param project_name: the name of the project
#         :return: a dictionary of FOVs in project
#         """
#         with pydetecdiv_project(project_name) as project:
#             return {cast(FOV, fov).name: cast(FOV, fov) for fov in project.get_objects('FOV')}
#
#     def accept(self) -> None:
#         """
#         Launch the drift computation and the associated waiting dialog
#         """
#         self.wait = WaitDialog('Computing drift, please wait.', self,
#                                cancel_msg='Cancel drift computation please wait')
#         self.finished.connect(self.wait.close_window)
#         self.wait.wait_for(self.compute_drift, Z=0, C=0)
#
#         tab = PyDetecDiv.main_window.add_tabbed_window(
#                 f'{PyDetecDiv.project_name} / Drift correction ({self.method_box.Method.value()})')
#         tab.project_name = PyDetecDiv.project_name
#         for fov in self.select_FOV.FOVs.selection():
#             tab.show_plot(self.drift[fov.name], title=fov.name)
#
#     def compute_drift(self, Z: int = 0, C: int = 0) -> None:
#         """
#         Compute the drift for the given Z and C as references
#
#         :param Z: the reference z-layer
#         :param C: the reference channel
#         """
#         for fov in self.select_FOV.FOVs.selection():
#             self.drift[fov.name] = fov.image_resource().image_resource_data().compute_drift(Z=Z, C=C,
#                                                                                             method=self.method_box.Method.value())
#         self.finished.emit(True)
#
#
# class ComputeDrift(QAction):
#     """
#     Action to compute drift correction
#     """
#
#     def __init__(self, parent: QWidget):
#         super().__init__("Compute drift", parent)
#         self.triggered.connect(self.open_dialog)
#         self.setEnabled(False)
#         parent.addAction(self)
#
#     def enable(self, project_name: str) -> None:
#         """
#         Enable or disable this action in the Data menu whether there are raw data or not.
#
#         :param project_name: the name of the project
#         """
#         if project_name:
#             with pydetecdiv_project(project_name) as project:
#                 if project.count_objects('FOV'):
#                     self.setEnabled(True)
#                 else:
#                     self.setEnabled(False)
#         else:
#             self.setEnabled(False)
#
#     def open_dialog(self) -> None:
#         """
#         Open the Compute-drift dialog window
#         """
#         _ = ComputeDriftDialog(title='Compute drift')
#

# class ApplyDrift(QAction):
#     """
#     Action to set or unset drift correction
#     """
#
#     def __init__(self, parent: QWidget):
#         super().__init__("Apply drift", parent)
#         # self.triggered.connect(self.)
#         self.setCheckable(True)
#         self.setChecked(False)
#         self.setEnabled(False)
#         parent.addAction(self)
#
#     def enable(self, project_name: str):
#         """
#         Enable or disable this action in the Data menu whether there are raw data or not.
#
#         :param project_name: the name of the project
#         """
#         if project_name:
#             with pydetecdiv_project(project_name) as project:
#                 if project.count_objects('FOV'):
#                     self.setEnabled(True)
#                 else:
#                     self.setEnabled(False)
#                 # # The following should allow to enable drift correction if there is drift information in the database
#                 # # but it may be quite time-consuming if there are many FOVs
#                 # if any([fov.image_resource().drift is not None for fov in project.get_objects('FOV')]):
#                 #     self.setEnabled(True)
#                 # else:
#                 #     self.setEnabled(False)
#         else:
#             self.setEnabled(False)


# class ConvertToNDTiffDialog(FileListChooserDialog):
#     """
#     A dialog window providing GUI for converting multiple images to ND-Tiff
#     """
#
#     def __init__(self):
#         super().__init__(title='Convert to NDTiff using metadata files', filters=["TXT (*.txt)", ], extensions=['*.txt'],
#                          destination=True)
#         # A reference to dataset should be kept to avoid destroying the object along with the wait dialog. Thus, internal threads
#         # can be closed properly when dataset is finished. Otherwise, finishing the dataset within the subthread might return an
#         # error.
#         self.dataset = None
#
#     def accept(self):
#         wait_dialog = WaitDialog('Converting multiple TIFF files to NDTiff', self,
#                                  cancel_msg='Rollback of NDTiff conversion: please wait', progress_bar=True, )
#         self.finished.connect(wait_dialog.close_window)
#         self.progress.connect(wait_dialog.show_progress)
#         wait_dialog.wait_for(self.conversion)
#         self.list_model.removeRows(0, self.list_model.rowCount())
#         self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
#         self.dataset.finish()
#
#     def conversion(self) -> None:
#         """
#         Converts multiple TIFF files to ND-Tiff
#         """
#         self.progress.emit(0)
#         # project_path = os.path.join(get_config_value('project', 'workspace'), PyDetecDiv.project_name)
#         ndtiff_path = './NDTiff' if self.destination.text() == '' else self.destination.text()
#         with open(self.file_list[0]) as f:
#             summary_metadata = json.load(f)['Summary']
#         if summary_metadata['Width'] == 0:
#             summary_metadata['Width'] = -1
#         if summary_metadata['Height'] == 0:
#             summary_metadata['Height'] = -1
#         self.dataset = NDTiffDataset(ndtiff_path, summary_metadata=summary_metadata, writable=True)
#         num_images = len([v for f in self.file_list for k, v in json.load(open(f)).items() if k.startswith('Metadata-')])
#         i = 0
#         for f in self.file_list:
#             print(f)
#             with open(f) as metadata_file:
#                 metadata = json.load(metadata_file)
#                 summary = metadata['Summary']
#                 for d in [v for k, v in metadata.items() if k.startswith('Metadata-')]:
#                     image_coordinates = {'channel' : d['ChannelIndex'], 'time': d['FrameIndex'], 'z': d['SliceIndex'],
#                                          'position': d['PositionIndex']
#                                          }
#                     pixels = tifffile.imread(os.path.join(os.path.dirname(f), os.path.basename(d["FileName"])))
#                     d['PositionName'] = summary['StagePositions'][d['PositionIndex']]['Label']
#                     self.dataset.put_image(image_coordinates, pixels, d)
#                     i = i + 100
#                     self.progress.emit(i / num_images)
#         # self.dataset.finish()
#         self.finished.emit(True)
#
#
# class ConvertToNDTiff(QAction):
#     """
#     Action for conversion of multiple image files to ND-Tiff
#     """
#
#     def __init__(self, parent: QWidget):
#         super().__init__("Convert to NDTiff", parent)
#         self.triggered.connect(ConvertToNDTiffDialog)
#         self.setEnabled(True)
#         parent.addAction(self)
#
#
# class ImportNDTiffDataDialog(gui.Dialog):
#     """
#     A Dialog window providing GUI for importing NDTiff data
#     """
#
#     def __init__(self, **kwargs: dict[str, Any]):
#         super().__init__(title='Import NDTiff data', **kwargs)
#
#         ndtiff_dir_box = self.addGroupBox('NDTiff directory', widget=gui.GroupBox)
#         ndtiff_layout = QHBoxLayout(ndtiff_dir_box)
#         self.ndtiff_dir = QLineEdit(ndtiff_dir_box, )
#         self.ndtiff_dir.setMinimumWidth(350)
#         button_path = QPushButton(ndtiff_dir_box)
#         button_path.setIcon(QIcon(":icons/file_chooser"))
#         button_path.clicked.connect(self.select_path)
#         ndtiff_layout.addWidget(self.ndtiff_dir)
#         ndtiff_layout.addWidget(button_path)
#
#         self.button_box = self.addButtonBox()
#         self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
#
#         self.arrangeWidgets([
#             ndtiff_dir_box,
#             self.button_box,
#             ])
#
#         gui.set_connections({self.button_box.accepted   : self.accept,
#                              self.button_box.rejected   : self.close,
#                              self.ndtiff_dir.textChanged: self.check_is_ndtiff,
#                              })
#
#         self.fit_to_contents()
#         self.exec()
#         for child in self.children():
#             child.deleteLater()
#         self.destroy(True)
#
#     def accept(self, /) -> None:
#         """
#         When Ok button is clicked, launch the NDTiff data import
#         """
#         with pydetecdiv_project(PyDetecDiv.project_name) as project:
#             project.import_ndtiff_data(self.ndtiff_dir.text())
#             PyDetecDiv.app.project_selected.emit(PyDetecDiv.project_name)
#             self.close()
#
#     def select_path(self) -> None:
#         """
#         Open a File dialog to select a path to NDTiff dataset
#         """
#         dir_name = '.'
#         if dir_name != self.ndtiff_dir.text() and self.ndtiff_dir.text():
#             dir_name = self.ndtiff_dir.text()
#         directory = QFileDialog.getExistingDirectory(self, caption='Choose data source directory', dir=dir_name,
#                                                      options=QFileDialog.Option.ShowDirsOnly)
#         if directory:
#             self.ndtiff_dir.setText(directory)
#
#     def check_is_ndtiff(self) -> None:
#         """
#         Check whether the specified path is a NDTiff path and enables the Ok button if it is
#         """
#         if self.ndtiff_dir.text() != '' and os.path.isfile(os.path.join(self.ndtiff_dir.text(), 'NDTiff.index')):
#             self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
#         else:
#             self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
#
#
# class ImportNDTiffData(QAction):
#     """
#     Action to import raw data images into a project
#     """
#
#     def __init__(self, parent: QWidget):
#         super().__init__(QIcon(":icons/import_images"), "Import NDTiff data", parent)
#         self.triggered.connect(ImportNDTiffDataDialog)
#         self.setEnabled(False)
#         parent.addAction(self)
#
#
# class ImportROIannotations(QAction):
#     """
#     Action to import annotated ROIs into a project
#     """
#     def __init__(self, parent: QWidget):
#         super().__init__("Import annotated ROIs", parent)
#         self.triggered.connect(self.import_annotated_rois)
#         self.setEnabled(False)
#         parent.addAction(self)
#
#     def import_annotated_rois(self) -> None:
#         """
#         Import a csv file containing annotated ROIs. ROIs that do not exist yet in the project are created.
#         """
#         filters = ["csv (*.csv)", "tsv (*.tsv)", ]
#         annotation_file, _ = QFileDialog.getOpenFileName(PyDetecDiv.main_window,
#                                                          caption='Choose file with annotated ROIs',
#                                                          dir='.',
#                                                          filter=";;".join(filters),
#                                                          selectedFilter=filters[0])
#         if annotation_file:
#             print('Import annotated ROIs from file')
#             with pydetecdiv_project(PyDetecDiv.project_name) as project:
#                 fov_list = {cast(FOV, fov).name: fov.id_ for fov in project.get_objects('FOV')}
#                 roi_names = [cast(ROI, roi).name for roi in project.get_objects('ROI')]
#                 annotated_rois = polars.read_csv(annotation_file).filter(polars.col('fov').is_in(fov_list))
#                 classification = cast(Classification, project.get_object('Classification', 1))
#                 class_names = annotated_rois.select('class_name').unique('class_name').to_numpy().flatten()
#                 print(class_names, classification.classes)
#                 if not set(class_names).issubset(set(classification.classes)):
#                     print('Invalid class names: not compatible with the current classification scheme.')
#                     return
#                 run = Run(project=project, tool_name='ROI annotations', tool_version='1.0.0',
#                           command='import_annotated_rois')
#                 for row in annotated_rois.iter_rows(named=True):
#                     if row['roi'] not in roi_names:
#                         new_roi = ROI(project=project, name=row['roi'], fov=fov_list[row['fov']],
#                                       top_left=(row['x'], row['y']),
#                                       bottom_right=(row['x'] + row['width'], row['y'] + row['height']))
#                         roi_names.append(new_roi.name)
#                     _ = RoiAnnotations(project=project, roi=new_roi.id_, t=row['frame'],
#                                        classification=classification, annotation=row['class_name'],
#                                        run=run, key_val={'class_name': row['class_name']})
#                 project.commit()
#             print(annotated_rois)
