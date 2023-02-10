"""
Handling actions to open, create and interact with projects
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QFileDialog)
from pydetecdiv.app import PyDetecDivApplication, get_settings, WaitDialog, PyDetecDivThread, pydetecdiv_project


class ImportDataDialog(QFileDialog):
    """
    A dialog window to edit settings
    """

    def __init__(self, choose_dir=False):
        super().__init__(PyDetecDivApplication.main_window)
        self.settings = get_settings()
        self.setWindowModality(Qt.WindowModal)

        if choose_dir:
            self.setFileMode(QFileDialog.Directory)
        else:
            self.setFileMode(QFileDialog.ExistingFiles)

        self.setNameFilters(["TIFF (*.tif *.tiff)",
                             "JPEG (*.jpg *.jpeg)",
                             "PNG (*.png)",
                             "Image files (*.tif *.tiff, *.jpg *.jpeg, *.png)",])

        self.filesSelected.connect(self.import_files)

        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy()

    def import_files(self):
        with pydetecdiv_project(PyDetecDivApplication.project_name) as project:
            print(self.selectedFiles())
            # project.import_images(f'{self.selectedFiles()[0]}/*')


class ImportData(QAction):
    def __init__(self, icon, label, parent, choose_dir=False):
        super().__init__(icon, label, parent)
        self.triggered.connect(lambda _: ImportDataDialog(choose_dir=choose_dir))
        self.setEnabled(False)
        parent.addAction(self)


class ImportDataFiles(ImportData):
    """
    Action to import raw data images into a project
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/import_images"), "&Import image files", parent, choose_dir=False)

class ImportDataDir(ImportData):
    """
    Action to import raw data images into a project
    """

    def __init__(self, parent):
        super().__init__(QIcon(":icons/import_folder"), "Import &directory", parent, choose_dir=True)
