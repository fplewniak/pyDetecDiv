"""
Handling actions to open, create and interact with projects
"""
import glob
import os
import time
from pathlib import Path

from PySide6.QtCore import Qt, QRegularExpression, QStringListModel, QItemSelectionModel, QItemSelection, Signal, QDir
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QFileDialog, QDialog, QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QDialogButtonBox, QListView, QComboBox, QMenu, QAbstractItemView)
from pydetecdiv.app import PyDetecDiv, get_settings, WaitDialog, pydetecdiv_project


class ListView(QListView):
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


class ImportDataDialog(QDialog):
    """
    A dialog window to choose sources for image data files to import into the project raw dataset
    """
    progress = Signal(int)
    chosen_directory = Signal(str)

    def __init__(self):
        super().__init__(PyDetecDiv().main_window)
        settings = get_settings()
        self.project_path = os.path.join(settings.value("project/workspace"), PyDetecDiv().project_name)
        self.setWindowModality(Qt.WindowModal)
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
        self.default_extension.addItems(['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*',])

        list_view = ListView(source_group_box)
        self.list_model = QStringListModel()
        list_view.setModel(self.list_model)

        destination_widget = QGroupBox(self)
        destination_widget.setTitle(f'Destination: {self.project_path}/data/')
        self.destination_directory = QComboBox(destination_widget)
        self.destination_directory.addItems(self.get_destinations())
        self.destination_directory.setEditable(True)
        self.destination_directory.setValidator(self.sub_directory_name_validator())

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close | QDialogButtonBox.Cancel | QDialogButtonBox.Ok, self)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

        add_path_dialog = AddPathDialog(self)
        # Layout
        vertical_layout = QVBoxLayout(self)
        source_layout = QVBoxLayout(source_group_box)
        buttons_layout = QHBoxLayout(buttons_widget)
        extension_layout = QHBoxLayout(extension_widget)
        destination_layout = QHBoxLayout(destination_widget)

        source_layout.addWidget(list_view)
        source_layout.addWidget(buttons_widget)
        source_layout.addWidget(extension_widget)

        buttons_layout.addWidget(path_button)
        buttons_layout.addWidget(directory_button)
        buttons_layout.addWidget(files_button)

        extension_layout.addWidget(extension_label)
        extension_layout.addWidget(self.default_extension)

        destination_layout.addWidget(self.destination_directory)

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
        self.chosen_directory.connect(lambda path: add_path_dialog.set_path(path))

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
                                                filter= ";;".join(filters),
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
            file_list += [f for f in glob.glob(source_path) if os.path.isfile(f)]
        return file_list

    def accept(self):
        """
        Import files whose list is defined by the sources in self.list_model
        """
        WaitDialog(f'Importing data into {PyDetecDiv().project_name}', self.import_data, self, hide_parent=False,
                   progress_bar=True, n_max=len(self.file_list()))
        self.list_model.removeRows(0, self.list_model.rowCount())
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

    def import_data(self):
        """
        Import image data from source specified in list_model into project raw dataset and triggers a progress signal
        with the number of files that have been copied so far
        """
        destination = os.path.join(self.project_path, 'data', self.destination_directory.currentText())
        print(destination)
        QDir().mkpath(str(destination))
        n_start = len(os.listdir(destination))
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            for source_path in self.list_model.stringList():
                project.import_images(source_path, destination=self.destination_directory.currentText())
                n = len(os.listdir(destination)) - n_start
                self.progress.emit(n)
        while n < len(self.file_list()):
            n = len(os.listdir(destination)) - n_start
            self.progress.emit(n)

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
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self.button_box.button(QDialogButtonBox.Apply).setEnabled(False)
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
        self.path_text_input.textChanged.connect(self.path_specification_changed)

    def accept(self):
        """
        Accept the path input text and add it to the source list
        """
        self.path_validated.emit(self.path_text_input.text())
        self.hide()

    def set_path(self, path):
        self.path_text_input.setText(path)

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
