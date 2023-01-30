from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication
from pydetecdiv.persistence.project import list_projects
from pydetecdiv.app import get_settings


class OpenProject(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&Open project", parent)
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/folder-open.png"))
        self.triggered.connect(self.open_project)
        parent.addAction(self)

    @Slot()
    def open_project(self):
        print(list_projects())


class Settings(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&Settings", parent)
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/gear.png"))
        self.triggered.connect(self.edit_settings)
        parent.addAction(self)

    @Slot()
    def edit_settings(self):
        settings = get_settings()
        for k in settings.allKeys():
            print(f'{k}: {settings.value(k)}')
        print(QApplication.topLevelWindows())

class Quit(QAction):
    def __init__(self, parent, icon=False):
        super().__init__("&Quit", parent)
        if icon:
            self.setIcon(QIcon("/home/fred/PycharmProjects/fugue-icons-3.5.6-src/icons/door-open-out.png"))
        self.triggered.connect(self.quit_app)
        parent.addAction(self)

    @Slot()
    def quit_app(self):
        QApplication.quit()
