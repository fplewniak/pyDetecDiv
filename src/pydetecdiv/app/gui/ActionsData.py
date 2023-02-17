"""
Handling actions to open, create and interact with projects
"""
import glob
import itertools
import os

from PySide6.QtCore import Qt, QRegularExpression, QStringListModel, QItemSelectionModel, QItemSelection
from PySide6.QtGui import QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QFileDialog, QDialog, QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QDialogButtonBox, QListView, QComboBox, QMenu, QAbstractItemView)
from pydetecdiv.app import PyDetecDivApplication, get_settings, WaitDialog, PyDetecDivThread, pydetecdiv_project


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
    def __init__(self):
        super().__init__(PyDetecDivApplication.main_window)
        settings = get_settings()
        self.project_path = os.path.join(settings.value("project/workspace"), PyDetecDivApplication.project_name)
        self.setWindowModality(Qt.WindowModal)

        self.setObjectName('ImportData')
        self.setWindowTitle('Import image data')

        # Widgets
        source_group_box = QGroupBox(self)
        source_group_box.setTitle('Source for image files to import:')

        buttons_widget = QWidget(source_group_box)
        files_button = QPushButton('Add files', buttons_widget)
        directory_button = QPushButton('Add directory', buttons_widget)
        path_button = QPushButton('Add path', buttons_widget)
        extension_widget = QWidget(source_group_box)
        extension_label = QLabel('Default image file extension:', extension_widget)
        self.default_extension = QComboBox(extension_widget)
        self.default_extension.addItems(['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png'])

        # self.list_view = QListView(self.source_group_box)
        list_view = ListView(source_group_box)
        self.list_model = QStringListModel()
        list_view.setModel(self.list_model)

        destination_widget = QGroupBox(self)
        destination_widget.setTitle(f'Destination: {self.project_path}/data/')
        destination_directory = QComboBox(destination_widget)
        destination_directory.addItems(self.get_destinations())
        destination_directory.setEditable(True)
        destination_directory.setValidator(self.sub_directory_name_validator())

        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok, self)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

        # Layout
        vertical_layout = QVBoxLayout(self)
        source_layout = QVBoxLayout(source_group_box)
        buttons_layout = QHBoxLayout(buttons_widget)
        extension_layout = QHBoxLayout(extension_widget)
        destination_layout = QHBoxLayout(destination_widget)

        source_layout.addWidget(list_view)
        source_layout.addWidget(buttons_widget)
        source_layout.addWidget(extension_widget)

        buttons_layout.addWidget(files_button)
        buttons_layout.addWidget(directory_button)
        buttons_layout.addWidget(path_button)

        extension_layout.addWidget(extension_label)
        extension_layout.addWidget(self.default_extension)

        destination_layout.addWidget(destination_directory)

        vertical_layout.addWidget(source_group_box)
        vertical_layout.addWidget(destination_widget)
        vertical_layout.addWidget(self.button_box)

        # Widgets behaviour
        #
        files_button.clicked.connect(
            lambda _: AddFilesDialog(self, model=self.list_model, choose_dir=False))
        directory_button.clicked.connect(
            lambda _: AddFilesDialog(self, model=self.list_model, choose_dir=True))
        path_button.clicked.connect(lambda _: AddPathDialog(self, model=self.list_model))

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)

        self.list_model.dataChanged.connect(self.source_list_is_not_empty)

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy(True)

    def get_destinations(self):
        """
        Get the list of subdirectories in the destination raw dataset directory
        :return: list of subdirectories in the destination raw dataset directory
        :rtype: list of str
        """
        return [''] + [d.name for d in os.scandir(os.path.join(self.project_path, 'data')) if d.is_dir()]

    def accept(self):
        """
        Import files whose list is defined by the sources in self.list_model
        """
        selection_list = list(itertools.chain.from_iterable([glob.glob(p) for p in self.list_model.stringList()]))
        file_list = [f for f in selection_list if os.path.isfile(f)]
        print(f'Importing files {file_list}')
        self.close()

    def source_list_is_not_empty(self):
        """
        Checks the source list is not empty, enables OK button if not
        """
        if self.list_model.stringList():
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


class AddFilesDialog(QFileDialog):
    """
    A dialog window to select files or directory to add to import list
    """

    def __init__(self, parent_window, model, choose_dir=False):
        super().__init__(parent_window)
        self.setWindowModality(Qt.WindowModal)
        self.model = model
        self.choose_dir = choose_dir

        if choose_dir:
            self.setFileMode(QFileDialog.Directory)
        else:
            self.setFileMode(QFileDialog.ExistingFiles)
            self.setNameFilters(["TIFF (*.tif *.tiff)",
                                 "JPEG (*.jpg *.jpeg)",
                                 "PNG (*.png)",
                                 "Image files (*.tif *.tiff, *.jpg *.jpeg, *.png)", ])

        self.filesSelected.connect(self.select_files)

        self.exec()
        self.destroy(True)

    def select_files(self):
        """
        Add the selected files or directory to the list_model
        """
        if self.choose_dir:
            selected_files = [f'{self.selectedFiles()[0]}/{self.parent().default_extension.currentText()}']
        else:
            selected_files = self.selectedFiles()
        source_list = self.model.stringList()
        self.model.setStringList(source_list + selected_files)
        self.parent().button_box.button(QDialogButtonBox.Ok).setEnabled(True)


class AddPathDialog(QDialog):
    """
    A dialog window to select a path pointing to files or directories to import
    """

    def __init__(self, parent_window, model):
        super().__init__(parent_window)
        self.setWindowModality(Qt.WindowModal)
        self.model = model

        self.path_widget = QWidget(self)
        self.path_widget.setMinimumWidth(350)
        self.path_label = QLabel('Path:', self.path_widget)
        self.path_text_input = QLineEdit(self.path_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                           Qt.Horizontal, self.path_widget)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self.button_box.button(QDialogButtonBox.Apply).setEnabled(False)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self.add_path)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.path_widget)
        self.layout.addWidget(self.button_box)
        self.path_layout = QHBoxLayout(self.path_widget)
        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path_text_input)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)
        self.path_text_input.textChanged.connect(self.path_specification_changed)

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy(True)

    def accept(self):
        """
        Accept the path input text and add it to the source list
        """
        self.add_path()
        self.close()

    def add_path(self):
        """
        Add the path input text to the model as a new source. Adds a wildcard if the path points to directories
        """
        path_text = self.path_text_input.text()
        if os.path.isdir(glob.glob(path_text)[0]):
            path_text = os.path.join(path_text, self.parent().default_extension.currentText())
        source_list = self.model.stringList()
        self.model.setStringList(source_list + [path_text])
        self.parent().button_box.button(QDialogButtonBox.Ok).setEnabled(True)

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
