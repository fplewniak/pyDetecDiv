"""
Generic classes to discover and handle plugins
"""
import importlib
import os
import pkgutil
import sys

from PySide6.QtGui import QAction

import pydetecdiv

class Plugin:
    """
    Generic class defining common Plugin attributes and methods
    """
    id_ = None
    version = None
    name = None
    category = None

    def __init__(self):
        self.gui = None

    def addActions(self, menu):
        """
        Method to add an action to a menu. This action triggers the launch method. If a submenu needs to be implemented
        or if arguments need to be passed to the launch method, this method may be overriden
        :param menu:
        """
        action = QAction(self.name, menu)
        action.triggered.connect(self.launch)
        menu.addAction(action)

    def launch(self):
        """
        Abstract method that needs to be implemented in each concrete Plugin implementation to launch the plugin (with
        or without a GUI)
        """
        raise NotImplementedError


def get_plugins_dir():
    """
Get the user directory where plugins are installed. The directory is created if it does not exist
:return: the user plugin path
:rtype: Path
"""
    plugins_path = os.path.join(pydetecdiv.app.get_appdata_dir(), 'plugins')
    if not os.path.exists(plugins_path):
        os.mkdir(plugins_path)
    return [plugins_path]
    # return [pydetecdiv.plugins.__path__[0], plugins_path]


class PluginList:
    """
    Class to create and handle list of discovered plugins
    """

    def __init__(self):
        self.categories = []
        self.plugins_dict = {}

    @property
    def plugins(self):
        return self.plugins_dict.values()

    def load(self):
        """
        Discover plugins and load them in plugin list
        """
        # for _, name, _ in pkgutil.iter_modules(pydetecdiv.plugins.__path__):
        #     module = importlib.import_module(f'pydetecdiv.plugins.{name}')
        #     if module.Plugin.category not in self.categories:
        #         self.categories.append(module.Plugin.category)
        #     self.plugins.append(module.Plugin())
        # plugins_dir = pydetecdiv.plugins.__path__+[pydetecdiv.app.get_plugins_dir()]
        for finder, name, _ in pkgutil.iter_modules(pydetecdiv.plugins.__path__):
            loader = finder.find_module(name)
            spec = importlib.util.spec_from_file_location(f'pydetecdiv.plugins.{name}', loader.path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            if module.Plugin.category not in self.categories:
                self.categories.append(module.Plugin.category)
            # self.plugins.append(module.Plugin())
            self.plugins_dict[module.Plugin.id_] = module.Plugin()
        for finder, name, _ in pkgutil.iter_modules(get_plugins_dir()):
            loader = finder.find_module(name)
            spec = importlib.util.spec_from_file_location(name, loader.path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            if module.Plugin.category not in self.categories:
                self.categories.append(module.Plugin.category)
            # self.plugins.append(module.Plugin())
            self.plugins_dict[module.Plugin.id_] = module.Plugin()

    @property
    def len(self):
        """
        The number of discovered plugins
        :return: how many plugins have been discovered
        :rtype: int
        """
        return len(self.plugins)
