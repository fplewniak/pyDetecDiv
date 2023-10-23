from PySide6.QtCore import Qt
from PySide6.QtGui import QPen

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.Windows import TabbedViewer
from pydetecdiv.plugins.test.Actions import Action1, Action2

class Plugin(plugins.Plugin):
    name = 'Test'
    category = 'Tests'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f'Creating {self.name} plugin')
        # self.registered_objects = {
        #     'MainWindow': [],
        #     'TabbedViewer': [],
        # }

    # def register_object(self, obj):
    #     match obj.__class__.__name__:
    #         case 'MainWindow':
    #             print('Modify main window gui')
    #             self.registered_objects['MainWindow'].append(obj)
    #         case 'TabbedViewer':
    #             print(f'Modify tabbed viewer {obj.windowTitle()} gui')
    #             self.registered_objects['TabbedViewer'].append(obj)
    #             # obj.viewer.scene.pen = QPen(Qt.GlobalColor.blue, 6)
    #         case _:
    #             print(f'Do nothing with object of class {obj.__class__.__name__}')

    def addActions(self, menu):
        Action1(menu, self)
        Action2(menu, self)

    def change_pen(self):
        active_subwindow = PyDetecDiv().main_window.mdi_area.activeSubWindow()
        if active_subwindow:
            tab = [tab for tab in PyDetecDiv().main_window.tabs.values() if tab.window == active_subwindow][0]
            print(f'change pen in {tab.windowTitle()}')
            if tab.viewer.scene.pen.width() == 2:
                tab.viewer.scene.pen = QPen(Qt.GlobalColor.blue, 6)
            else:
                tab.viewer.scene.pen = QPen(Qt.GlobalColor.cyan, 2)


