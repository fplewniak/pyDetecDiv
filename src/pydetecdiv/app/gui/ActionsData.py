"""
Handling actions to open, create and interact with projects
"""
import glob
import json
import os

import numpy as np
from PySide6.QtCore import (Qt, QRegularExpression, QStringListModel, QItemSelectionModel, QItemSelection, Signal, QDir,
                            QThread)
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QFileDialog, QDialog, QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QDialogButtonBox, QListView, QComboBox, QMenu, QAbstractItemView,
                               QRadioButton, QButtonGroup)
from pydetecdiv.app import PyDetecDiv, WaitDialog, pydetecdiv_project, MessageDialog
from pydetecdiv.plugins.parameters import Parameter, ChoiceParameter

from pydetecdiv.settings import get_config_value
from pydetecdiv import delete_files
from pydetecdiv.app.gui.RawData2FOV import RawData2FOV
import pydetecdiv.plugins.gui as gui


class FileListView(QListView):
    """
    A class extending QListView to display source for image data. Defines a context menu to clear or toggle selection,
    remove selected sources, clear list
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.MultiSelection)

    def contextMenuEvent(self, e):
        """
        Definition of a context menu to clear or toggle selection of sources in list model, remove selected sources from
        the list model, clear the source list model

        :param e: mouse event providing the position of the context menu
        :type e: PySide6.QtGui.QContextMenuEvent
        """
        if self.model().rowCount():
            context = QMenu(self)
            unselect = QAction("Unselect all", self)
            unselect.triggered.connect(self.unselect)
            context.addAction(unselect)
            toggle = QAction("Toggle selection", self)
            toggle.triggered.connect(self.toggle)
            context.addAction(toggle)
            context.addSeparator()
            remove = QAction("Remove selected items", self)
            remove.triggered.connect(self.remove_items)
            context.addAction(remove)
            clear_list = QAction("Clear list", self)
            context.addAction(clear_list)
            clear_list.triggered.connect(self.clear_list)
            context.exec(e.globalPos())

    def unselect(self):
        """
        Clear selection model
        """
        self.selectionModel().clear()

    def toggle(self):
        """
        Toggle selection model, selected sources are deselected and unselected ones are selected
        """
        toggle_selection = QItemSelection()
        top_left = self.model().index(0, 0)
        bottom_right = self.model().index(self.model().rowCount() - 1, 0)
        toggle_selection.select(top_left, bottom_right)
        self.selectionModel().select(toggle_selection, QItemSelectionModel.Toggle)

    def remove_items(self):
        """
        Delete selected sources
        """
        for idx in sorted(self.selectedIndexes(), key=lambda x: x.row(), reverse=True):
            self.model().removeRow(idx.row())

    def clear_list(self):
        """
        Clear the source list
        """
        self.model().removeRows(0, self.model().rowCount())


class ImportMetaDataDialog(QDialog):
    """
    A dialog window to choose sources for metadata files to import images and create Image resources
    """
    progress = Signal(int)
    chosen_directory = Signal(str)
    finished = Signal(bool)

    def __init__(self):
        super().__init__(PyDetecDiv.main_window)
        self.project_path = os.path.join(get_config_value('project', 'workspace'), PyDetecDiv.project_name)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(450)
        self.current_dir = '.'

        self.setObjectName('ImportMetaData')
        self.setWindowTitle('Import image data from metadata')

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close | QDialogButtonBox.Cancel | QDialogButtonBox.Ok, self)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

        source_group_box = QGroupBox(self)
        source_group_box.setTitle('Metadata files:')

        buttons_widget = QWidget(source_group_box)
        directory_button = QPushButton('Add directory', buttons_widget)
        path_button = QPushButton('Add path', buttons_widget)
        files_button = QPushButton('Add files', buttons_widget)
        extension_widget = QWidget(source_group_box)
        extension_label = QLabel('Default metadata file extension:', extension_widget)
        self.default_extension = QComboBox(extension_widget)
        self.default_extension.addItems(['*.txt', '*.json', '*', ])

        list_view = FileListView(source_group_box)
        self.list_model = QStringListModel()
        list_view.setModel(self.list_model)

        add_path_dialog = AddPathDialog(self)

        vertical_layout = QVBoxLayout(self)
        source_layout = QVBoxLayout(source_group_box)
        buttons_layout = QHBoxLayout(buttons_widget)
        extension_layout = QHBoxLayout(extension_widget)

        source_layout.addWidget(list_view)
        source_layout.addWidget(buttons_widget)
        source_layout.addWidget(extension_widget)

        buttons_layout.addWidget(path_button)
        buttons_layout.addWidget(directory_button)
        buttons_layout.addWidget(files_button)

        source_layout.addWidget(list_view)
        source_layout.addWidget(buttons_widget)
        source_layout.addWidget(extension_widget)

        buttons_layout.addWidget(path_button)
        buttons_layout.addWidget(directory_button)
        buttons_layout.addWidget(files_button)

        extension_layout.addWidget(extension_label)
        extension_layout.addWidget(self.default_extension)

        vertical_layout.addWidget(source_group_box)

        vertical_layout.addWidget(self.button_box)

        files_button.clicked.connect(self.add_files)
        directory_button.clicked.connect(self.add_dir)
        path_button.clicked.connect(add_path_dialog.show)
        add_path_dialog.path_validated.connect(self.add_path)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy(True)

    def add_files(self):
        """
        Open a file chooser dialog box and add selected files to the source model
        """
        filters = ["TXT (*.txt)",
                   "JSON (*.json)"]
        files, _ = QFileDialog.getOpenFileNames(self, caption='Choose metadata files',
                                                dir=self.current_dir,
                                                filter=";;".join(filters),
                                                selectedFilter=filters[0])
        if files:
            self.current_dir = os.path.dirname(files[0])
            self.list_model.setStringList(self.list_model.stringList() + files)
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def add_dir(self):
        """
        Open a directory chooser dialog box and add selected directory to the source model
        """
        directory = QFileDialog.getExistingDirectory(self, caption='Choose metadata directory', dir=self.current_dir,
                                                     options=QFileDialog.ShowDirsOnly)
        if directory:
            self.current_dir = directory
            self.chosen_directory.emit(str(os.path.join(directory, self.default_extension.currentText())))
            self.list_model.setStringList(self.list_model.stringList()
                                          + [os.path.join(directory, self.default_extension.currentText())])
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def add_path(self, path):
        """
        Add the input path to the source model

        :param path:
        """
        self.list_model.setStringList(self.list_model.stringList() + [path])
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def accept(self):
        """
        Import files whose list is defined by the sources in self.list_model
        """
        wait_dialog = WaitDialog(f'Importing data into {PyDetecDiv.project_name}', self,
                                 cancel_msg='Rollback of image import: please wait', progress_bar=True, )
        self.finished.connect(wait_dialog.close_window)
        self.progress.connect(wait_dialog.show_progress)
        wait_dialog.wait_for(self.import_data)
        self.list_model.removeRows(0, self.list_model.rowCount())
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

    def import_data(self):
        # destination = os.path.join(self.project_path, 'data', self.destination_directory.currentText())
        self.progress.emit(0)
        i = 0.0
        metadata_file_names = [f for source_path in self.list_model.stringList() for f in glob.glob(source_path) if
                               os.path.isfile(f)]
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            for metadata_file_name in metadata_file_names:
                i += 1.0
                project.import_images_from_metadata(metadata_file_name)
                self.progress.emit(100 * i / len(metadata_file_names))
        project.commit()
        self.finished.emit(True)
        PyDetecDiv.app.project_selected.emit(PyDetecDiv.project_name)


class ImportDataDialog(QDialog):
    """
    A dialog window to choose sources for image data files to import into the project raw dataset
    """
    progress = Signal(int)
    chosen_directory = Signal(str)
    finished = Signal(bool)

    def __init__(self):
        super().__init__(PyDetecDiv.main_window)
        self.project_path = os.path.join(get_config_value('project', 'workspace'), PyDetecDiv.project_name)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(450)
        self.current_dir = '.'

        self.setObjectName('ImportData')
        self.setWindowTitle('Import image data')

        # Widgets
        source_group_box = QGroupBox(self)
        source_group_box.setTitle('Source for image files to import:')

        buttons_widget = QWidget(source_group_box)
        directory_button = QPushButton('Add directory', buttons_widget)
        path_button = QPushButton('Add path', buttons_widget)
        files_button = QPushButton('Add files', buttons_widget)
        extension_widget = QWidget(source_group_box)
        extension_label = QLabel('Default image file extension:', extension_widget)
        self.default_extension = QComboBox(extension_widget)
        self.default_extension.addItems(['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.txt', '*', ])

        list_view = FileListView(source_group_box)
        self.list_model = QStringListModel()
        list_view.setModel(self.list_model)

        destination_widget = QGroupBox(self)
        destination_widget.setTitle('Destination:')
        copy_files_widget = QWidget(destination_widget)
        copy_files_button = QRadioButton(f'{self.project_path}/data/', copy_files_widget)
        self.destination_directory = QComboBox(copy_files_widget)
        self.destination_directory.addItems(self.get_destinations())
        self.destination_directory.setEditable(True)
        self.destination_directory.setValidator(self.sub_directory_name_validator())
        keep_in_place_widget = QWidget(destination_widget)
        keep_in_place_button = QRadioButton('keep files in place', keep_in_place_widget)
        self.keep_copy_buttons = QButtonGroup(destination_widget)
        self.keep_copy_buttons.addButton(copy_files_button, id=1)
        self.keep_copy_buttons.addButton(keep_in_place_button, id=2)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close | QDialogButtonBox.Cancel | QDialogButtonBox.Ok, self)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

        add_path_dialog = AddPathDialog(self)
        # Layout
        vertical_layout = QVBoxLayout(self)
        source_layout = QVBoxLayout(source_group_box)
        buttons_layout = QHBoxLayout(buttons_widget)
        extension_layout = QHBoxLayout(extension_widget)
        copy_files_layout = QHBoxLayout(copy_files_widget)
        keep_in_place_layout = QHBoxLayout(keep_in_place_widget)
        destination_layout = QVBoxLayout(destination_widget)

        source_layout.addWidget(list_view)
        source_layout.addWidget(buttons_widget)
        source_layout.addWidget(extension_widget)

        buttons_layout.addWidget(path_button)
        buttons_layout.addWidget(directory_button)
        buttons_layout.addWidget(files_button)

        extension_layout.addWidget(extension_label)
        extension_layout.addWidget(self.default_extension)

        copy_files_layout.addWidget(copy_files_button)
        copy_files_layout.addWidget(self.destination_directory)
        keep_in_place_layout.addWidget(keep_in_place_button)
        destination_layout.addWidget(keep_in_place_widget)
        destination_layout.addWidget(copy_files_widget)

        vertical_layout.addWidget(source_group_box)
        vertical_layout.addWidget(destination_widget)
        vertical_layout.addWidget(self.button_box)

        # Widgets behaviour
        files_button.clicked.connect(self.add_files)
        directory_button.clicked.connect(self.add_dir)
        path_button.clicked.connect(add_path_dialog.show)
        add_path_dialog.path_validated.connect(self.add_path)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)

        self.list_model.dataChanged.connect(self.source_list_is_not_empty)
        self.chosen_directory.connect(add_path_dialog.path_text_input.setText)

        keep_in_place_button.setChecked(True)
        self.keep_copy_buttons.setExclusive(True)

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy(True)

    def add_files(self):
        """
        Open a file chooser dialog box and add selected files to the source model
        """
        filters = ["TIFF (*.tif *.tiff)",
                   "JPEG (*.jpg *.jpeg)",
                   "PNG (*.png)",
                   "Image files (*.tif *.tiff, *.jpg *.jpeg, *.png)"]
        files, _ = QFileDialog.getOpenFileNames(self, caption='Choose source files',
                                                dir=self.current_dir,
                                                filter=";;".join(filters),
                                                selectedFilter=filters[0])
        if files:
            self.current_dir = os.path.dirname(files[0])
            self.list_model.setStringList(self.list_model.stringList() + files)
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def add_dir(self):
        """
        Open a directory chooser dialog box and add selected directory to the source model
        """
        directory = QFileDialog.getExistingDirectory(self, caption='Choose source directory', dir=self.current_dir,
                                                     options=QFileDialog.ShowDirsOnly)
        if directory:
            self.current_dir = directory
            self.chosen_directory.emit(str(os.path.join(directory, self.default_extension.currentText())))
            self.list_model.setStringList(self.list_model.stringList()
                                          + [os.path.join(directory, self.default_extension.currentText())])
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def add_path(self, path):
        """
        Add the input path to the source model

        :param path:
        """
        self.list_model.setStringList(self.list_model.stringList() + [path])
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def get_destinations(self):
        """
        Get the list of subdirectories in the destination raw dataset directory

        :return: list of subdirectories in the destination raw dataset directory
        :rtype: list of str
        """
        return [''] + [d.name for d in os.scandir(os.path.join(self.project_path, 'data')) if d.is_dir()]

    def file_list(self):
        """
        Expands all source specification to return a list of files to import

        :return: file name list
        :rtype: list of str
        """
        file_list = []
        for source_path in self.list_model.stringList():
            if source_path.endswith('.txt'):
                for metadata_file_name in [f for f in glob.glob(source_path) if os.path.isfile(f)]:
                    with open(metadata_file_name) as metadata_file:
                        metadata = json.load(metadata_file)
                        file_list += [v["FileName"] for k, v in metadata.items() if k.startswith('Metadata-')]
            else:
                file_list += [f for f in glob.glob(source_path) if os.path.isfile(f)]
        return file_list

    def accept(self):
        """
        Import files whose list is defined by the sources in self.list_model
        """
        wait_dialog = WaitDialog(f'Importing data into {PyDetecDiv.project_name}', self,
                                 cancel_msg='Rollback of image import: please wait', progress_bar=True, )
        self.finished.connect(wait_dialog.close_window)
        self.progress.connect(wait_dialog.show_progress)
        wait_dialog.wait_for(self.import_data)
        self.list_model.removeRows(0, self.list_model.rowCount())
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

    def import_data(self):
        """
        Import image data from source specified in list_model into project raw dataset and triggers a progress signal
        with the number of files that have been copied so far
        """
        self.progress.emit(0)
        in_place = self.keep_copy_buttons.button(2).isChecked()
        destination = os.path.join(self.project_path, 'data', self.destination_directory.currentText())
        QDir().mkpath(str(destination))
        file_list = self.file_list()
        if len(file_list) == 0:
            self.finished.emit(True)
            MessageDialog('No data file to import in specified directories')
        else:
            n_files_to_copy = 0 if in_place else len(file_list)
            with pydetecdiv_project(PyDetecDiv.project_name) as project:
                initial_files = {d.url for d in project.get_objects('Data')}
                n_files0 = sum(1 for item in os.listdir(destination) if os.path.isfile(os.path.join(destination, item)))
                n_dso0 = project.count_objects('Data')
                self.progress.emit(0)
                processes = []
                for batch in np.array_split(file_list,
                                            int(len(file_list) / int(get_config_value('project', 'batch'))) + 1):
                    if len(batch):
                        process = project.import_images(batch, in_place=in_place,
                                                        destination=self.destination_directory.currentText())
                        processes.append(process)
                        n_files = 0 if in_place else self.count_imported_files(destination, n_files0)
                        n_dso = project.count_objects('Data') - n_dso0
                        self.progress.emit(int(100 * (n_files + n_dso) / (len(file_list) + n_files_to_copy)))
                    if QThread.currentThread().isInterruptionRequested():
                        self.cancel_import(initial_files, n_files0, project, processes)
                        return
                while (n_files + n_dso) < (len(file_list) + n_files_to_copy):
                    if QThread.currentThread().isInterruptionRequested():
                        self.cancel_import(initial_files, n_files0, project, processes)
                        return
                    n_files = 0 if in_place else self.count_imported_files(destination, n_files0)
                    n_dso = project.count_objects('Data') - n_dso0
                    self.progress.emit(int(100 * (n_files + n_dso) / (len(file_list) + n_files_to_copy)))
                n_raw_data_files = project.count_objects('Data')
            self.finished.emit(True)
            PyDetecDiv.app.raw_data_counted.emit(n_raw_data_files)

    def count_imported_files(self, destination, n_start):
        """
        Count imported files in destination directory to assess progress

        :param destination: destination directory which files are imported into
        :type destination: str
        :param n_start: the number of files already in the destination directory before import
        :type n_start: int
        :return: the number of imported files
        :rtype: int
        """
        return sum(1 for item in os.listdir(destination) if os.path.isfile(os.path.join(destination, item))) - n_start

    def cancel_import(self, initial_files, n_files0, project, processes):
        """
        Manage cancellation of import. Terminate all copy processes before launching deletion of files that were already
        copied. Then cancel persistence operations on Data objects, and eventually stop the host thread.

        :param imported:
        :param project:
        :param processes:
        """
        self.progress.emit(0)
        in_place = self.keep_copy_buttons.button(2).isChecked()
        destination = os.path.join(self.project_path, 'data', self.destination_directory.currentText())
        n_max = self.count_imported_files(destination, n_files0)
        n_files = 0 if in_place else self.count_imported_files(destination, n_files0)
        if self.keep_copy_buttons.button(1).isChecked():
            for process in processes:
                process.terminate()
            while n_files != self.count_imported_files(destination, n_files0):
                n_files = self.count_imported_files(destination, n_files0)
            all_files = {os.path.join(destination, item) for item in os.listdir(destination) if
                         os.path.isfile(os.path.join(destination, item))}
            diff = list(all_files.difference(initial_files))
            imported_batches = np.array_split(diff, int(len(diff) / int(get_config_value('project', 'batch'))) + 1)
            for cancelled in imported_batches:
                delete_files(cancelled)
                n_files = 0 if in_place else self.count_imported_files(destination, n_files0)
                self.progress.emit(100 - int(100 * n_files / n_max))
        project.cancel()
        while n_files > 0:
            n_files = 0 if in_place else self.count_imported_files(destination, n_files0)
            self.progress.emit(100 - int(100 * n_files / n_max))
        self.finished.emit(True)

    def source_list_is_not_empty(self):
        """
        Checks the source list is not empty, enables OK button if not
        """
        if self.list_model.rowCount():
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

    @staticmethod
    def sub_directory_name_validator():
        """
        Name validator to filter invalid character in directory name

        :return: the validator
        :rtype: QRegularExpressionValidator
        """
        name_filter = QRegularExpression()
        name_filter.setPattern('\\w[\\w-]*')
        validator = QRegularExpressionValidator()
        validator.setRegularExpression(name_filter)
        return validator


class ImportData(QAction):
    """
    Action to import raw data images into a project
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/import_images"), "&Import image files", parent)
        self.triggered.connect(ImportDataDialog)
        self.setEnabled(False)
        parent.addAction(self)


class ImportMetaData(QAction):
    """
    Action to import raw data images into a project
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/import_images"), "&Import metadata files", parent)
        self.triggered.connect(ImportMetaDataDialog)
        self.setEnabled(False)
        parent.addAction(self)


class AddPathDialog(QDialog):
    """
    A dialog window to select a path pointing to files or directories to import
    """
    path_validated = Signal(str)

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.setWindowModality(Qt.WindowModal)

        self.path_widget = QWidget(self)
        self.path_widget.setMinimumWidth(350)
        self.path_label = QLabel('Path:', self.path_widget)
        self.path_text_input = QLineEdit(self.path_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                           Qt.Horizontal, self.path_widget)
        # self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        # self.button_box.button(QDialogButtonBox.Apply).setEnabled(False)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(
            lambda _: self.path_validated.emit(self.path_text_input.text()))

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.path_widget)
        self.layout.addWidget(self.button_box)
        self.path_layout = QHBoxLayout(self.path_widget)
        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path_text_input)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)
        # self.path_text_input.textChanged.connect(self.path_specification_changed)

    def accept(self):
        """
        Accept the path input text and add it to the source list
        """
        self.path_validated.emit(self.path_text_input.text())
        self.hide()

    def path_specification_changed(self):
        """
        Checks the path input text actually exists and enables Apply and OK buttons accordingly
        """
        if glob.glob(self.path_text_input.text()):
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
            self.button_box.button(QDialogButtonBox.Apply).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            self.button_box.button(QDialogButtonBox.Apply).setEnabled(False)


class CreateFOV(QAction):
    """
    Action to import raw data images into a project
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/import_images"), "Build &Image resources from raw data", parent)
        self.triggered.connect(RawData2FOV)
        self.setEnabled(False)
        parent.addAction(self)

    def enable(self, raw_data_count):
        """
        Enable or disable this action in the Data menu whether there are raw data or not.

        :param raw_data_count: the number of files in raw dataset
        """
        if PyDetecDiv.project_name and (raw_data_count > 0):
            self.setEnabled(True)
        else:
            self.setEnabled(False)


class ComputeDriftDialog(gui.Dialog):
    """
    A dialog window to run drift correction computations for a selection of FOVs
    """
    # progress = Signal(int)
    finished = Signal(bool)

    def __init__(self, title=None):
        super().__init__(title=title)

        self.drift = {}

        self.select_FOV = self.addGroupBox('Select FOV')
        self.fov_list = self.select_FOV.addOption(None, widget=gui.ListView,
                                                  parameter=ChoiceParameter(name='FOVs', label='FOV',
                                                                      items=self.update_fov_list(
                                                                          PyDetecDiv.project_name)),
                                                  multiselection=True, height=75)

        self.method_box = self.addGroupBox('Method')
        self.method = self.method_box.addOption(None, widget=gui.ComboBox,
                                                parameter=ChoiceParameter(name='Method', label='Method', default='vidstab',
                                                                    items={'vidstab': None, 'phase correlation': None})
                                                )

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([
            self.select_FOV,
            self.method_box,
            self.button_box
        ])

        gui.set_connections({self.button_box.accepted: self.accept,
                             self.button_box.rejected: self.close,
                             PyDetecDiv.app.project_selected: self.update_fov_list,
                             })

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy(True)

    def update_fov_list(self, project_name):
        with pydetecdiv_project(project_name) as project:
            return {fov.name: fov for fov in project.get_objects('FOV')}

    def accept(self):
        self.wait = WaitDialog('Computing drift, please wait.', self,
                               cancel_msg='Cancel drift computation please wait')
        self.finished.connect(self.wait.close_window)
        self.wait.wait_for(self.compute_drift, Z=0, C=0)

        tab = PyDetecDiv.main_window.add_tabbed_window(
            f'{PyDetecDiv.project_name} / Drift correction ({self.method.value()})')
        tab.project_name = PyDetecDiv.project_name
        for fov in self.fov_list.selection():
            tab.show_plot(self.drift[fov.name], title=fov.name)

    def compute_drift(self, Z=0, C=0):
        for fov in self.fov_list.selection():
            self.drift[fov.name] = fov.image_resource().image_resource_data().compute_drift(Z=Z, C=C,
                                                                                            method=self.method.value())
        self.finished.emit(True)


class ComputeDrift(QAction):
    """
    Action to compute drift correction
    """

    def __init__(self, parent):
        super().__init__("Compute drift", parent)
        self.triggered.connect(self.open_dialog)
        self.setEnabled(False)
        parent.addAction(self)

    def enable(self, project_name):
        """
        Enable or disable this action in the Data menu whether there are raw data or not.

        :param raw_data_count: the number of files in raw dataset
        """
        if project_name:
            with pydetecdiv_project(project_name) as project:
                if project.count_objects('FOV'):
                    self.setEnabled(True)
                else:
                    self.setEnabled(False)
        else:
            self.setEnabled(False)

    def open_dialog(self):
        gui = ComputeDriftDialog(title='Compute drift')


class ApplyDrift(QAction):
    """
    Action to set or unset drift correction
    """

    def __init__(self, parent):
        super().__init__("Apply drift", parent)
        # self.triggered.connect(self.)
        self.setCheckable(True)
        self.setChecked(False)
        self.setEnabled(False)
        parent.addAction(self)

    def enable(self, project_name):
        """
        Enable or disable this action in the Data menu whether there are raw data or not.

        :param raw_data_count: the number of files in raw dataset
        """
        if project_name:
            with pydetecdiv_project(project_name) as project:
                if project.count_objects('FOV'):
                    self.setEnabled(True)
                else:
                    self.setEnabled(False)
                # # The following should allow to enable drift correction if there is drift information in the database
                # # but it may be quite time-consuming if there are many FOVs
                # if any([fov.image_resource().drift is not None for fov in project.get_objects('FOV')]):
                #     self.setEnabled(True)
                # else:
                #     self.setEnabled(False)
        else:
            self.setEnabled(False)
