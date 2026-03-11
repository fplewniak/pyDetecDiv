import glob
import os

from PySide6.QtCore import QItemSelection, QItemSelectionModel, QStringListModel, Qt, Signal
from PySide6.QtGui import QContextMenuEvent, QAction, QIcon
from PySide6.QtWidgets import (QListView, QWidget, QAbstractItemView, QMenu, QFileDialog, QDialogButtonBox, QHBoxLayout,
                               QVBoxLayout,
                               QComboBox, QLabel, QPushButton, QGroupBox, QDialog, QLineEdit)

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.settings import get_config_value


class FileListView(QListView):
    """
    A class extending QListView to display source for image data. Defines a context menu to clear or toggle selection,
    remove selected sources, clear list
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

    def contextMenuEvent(self, e: QContextMenuEvent) -> None:
        """
        Definition of a context menu to clear or toggle selection of sources in list model, remove selected sources from
        the list model, clear the source list model

        :param e: mouse event providing the position of the context menu
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

    def unselect(self) -> None:
        """
        Clear selection model
        """
        self.selectionModel().clear()

    def toggle(self) -> None:
        """
        Toggle selection model, selected sources are deselected and unselected ones are selected
        """
        toggle_selection = QItemSelection()
        top_left = self.model().index(0, 0)
        bottom_right = self.model().index(self.model().rowCount() - 1, 0)
        toggle_selection.select(top_left, bottom_right)
        self.selectionModel().select(toggle_selection, QItemSelectionModel.SelectionFlag.Toggle)

    def remove_items(self) -> None:
        """
        Delete selected sources
        """
        for idx in sorted(self.selectedIndexes(), key=lambda x: x.row(), reverse=True):
            self.model().removeRow(idx.row())

    def clear_list(self) -> None:
        """
        Clear the source list
        """
        self.model().removeRows(0, self.model().rowCount())


class AddPathDialog(QDialog):
    """
    A dialog window to select a path pointing to files or directories to import
    """
    path_validated = Signal(str)

    def __init__(self, parent_window: QWidget):
        super().__init__(parent_window)
        self.setWindowModality(Qt.WindowModality.WindowModal)

        self.path_widget = QWidget(self)
        self.path_widget.setMinimumWidth(350)
        self.path_label = QLabel('Path:', self.path_widget)
        self.path_text_input = QLineEdit(self.path_widget)

        self.button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                Qt.Orientation.Horizontal, self.path_widget)
        # self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        # self.button_box.button(QDialogButtonBox.Apply).setEnabled(False)
        self.button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
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

    def accept(self) -> None:
        """
        Accept the path input text and add it to the source list
        """
        self.path_validated.emit(self.path_text_input.text())
        self.hide()

    def path_specification_changed(self) -> None:
        """
        Checks the path input text actually exists and enables Apply and OK buttons accordingly
        """
        if glob.glob(self.path_text_input.text()):
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
            self.button_box.button(QDialogButtonBox.StandardButton.Apply).setEnabled(True)
        else:
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            self.button_box.button(QDialogButtonBox.StandardButton.Apply).setEnabled(False)


class FileListChooserDialog(QDialog):
    """
    A dialog window to choose sources for metadata files to import images and create Image resources
    """
    progress = Signal(int)
    chosen_directory = Signal(str)
    finished = Signal(bool)

    def __init__(self, title=None, filters=None, extensions=None, destination=False):
        super().__init__(PyDetecDiv.main_window)
        # self.project_path = os.path.join(get_config_value('project', 'workspace'), PyDetecDiv.project_name)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(450)
        self.current_dir = '.'

        self.setWindowTitle(title)

        self.button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok,
                self)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

        source_group_box = QGroupBox(self)
        source_group_box.setTitle('Path list:')

        buttons_widget = QWidget(source_group_box)
        directory_button = QPushButton('Add directory', buttons_widget)
        path_button = QPushButton('Add path', buttons_widget)
        files_button = QPushButton('Add files', buttons_widget)
        extension_widget = QWidget(source_group_box)
        extension_label = QLabel('Default file extension:', extension_widget)
        self.default_extension = QComboBox(extension_widget)
        if extensions is not None:
            self.default_extension.addItems(extensions)
        else:
            self.default_extension.addItems(['*'])
        if filters is not None:
            self.filters = filters
        else:
            self.filters = ['*']

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

        extension_layout.addWidget(extension_label)
        extension_layout.addWidget(self.default_extension)

        vertical_layout.addWidget(source_group_box)
        if destination:
            destination_box = QGroupBox(self)
            destination_box.setTitle('Destination:')
            destination_layout = QHBoxLayout(destination_box)
            self.destination = QLineEdit(destination_box)
            button_path = QPushButton(destination_box)
            button_path.setIcon(QIcon(":icons/file_chooser"))
            button_path.clicked.connect(self.select_path)
            destination_layout.addWidget(self.destination)
            destination_layout.addWidget(button_path)
            vertical_layout.addWidget(destination_box)

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

    def add_files(self) -> None:
        """
        Open a file chooser dialog box and add selected files to the source model
        """
        # filters = ["TXT (*.txt)",
        #            "JSON (*.json)"]
        files, _ = QFileDialog.getOpenFileNames(self, caption='Choose files',
                                                dir=self.current_dir,
                                                filter=";;".join(self.filters),
                                                selectedFilter=self.filters[0])
        if files:
            self.current_dir = os.path.dirname(files[0])
            self.list_model.setStringList(self.list_model.stringList() + files)
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def add_dir(self) -> None:
        """
        Open a directory chooser dialog box and add selected directory to the source model
        """
        directory = QFileDialog.getExistingDirectory(self, caption='Choose directory', dir=self.current_dir,
                                                     options=QFileDialog.Option.ShowDirsOnly)
        if directory:
            self.current_dir = directory
            self.chosen_directory.emit(str(os.path.join(directory, self.default_extension.currentText())))
            self.list_model.setStringList(self.list_model.stringList()
                                          + [os.path.join(directory, self.default_extension.currentText())])
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def add_path(self, path: str) -> None:
        """
        Add the input path to the source model

        :param path: the metadata file path
        """
        self.list_model.setStringList(self.list_model.stringList() + [path])
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    @property
    def file_list(self):
        return [f for source_path in self.list_model.stringList() for f in glob.glob(source_path) if os.path.isfile(f)]

    def accept(self):
        ...

    def select_path(self):
        dir_name = './NDTiff'
        if dir_name != self.destination.text() and self.destination.text():
            dir_name = self.destination.text()
        directory = QFileDialog.getExistingDirectory(self, caption='Choose data source directory', dir=dir_name,
                                                     options=QFileDialog.Option.ShowDirsOnly)
        if directory:
            self.destination.setText(directory)
