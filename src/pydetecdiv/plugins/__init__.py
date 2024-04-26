"""
Generic classes to discover and handle plugins
"""
import importlib
import pkgutil
import sys

from PySide6.QtGui import QAction

import pydetecdiv
from pydetecdiv.settings import get_plugins_dir
from pydetecdiv.domain.Run import Run
from pydetecdiv.plugins.gui import Parameters, Dialog


class Plugin:
    """
    Generic class defining common Plugin attributes and methods
    """
    id_ = None
    version = None
    name = None
    category = None
    parent = None

    def __init__(self):
        self.gui = None
        self.parameters = Parameters()

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

    @property
    def parent_plugin(self):
        """
        return the parent plugin of the current plugin. This may be used to add functionalities to plugins without
        having to modify the original code

        :return: None or the parent plugin
        """
        if self.parent in pydetecdiv.app.PyDetecDiv().plugin_list.plugins_dict:
            return pydetecdiv.app.PyDetecDiv().plugin_list.plugins_dict[self.parent]
        return None

    def save_run(self, project, method, parameters):
        """
        Save the current Run

        :param project: the current project
        :param method: the plugin method that was executed for the current run
        :param parameters: the parameters that were passed to the method
        :return: the saved Run instance
        """
        record = {
            'tool_name': self.id_,
            'tool_version': self.version,
            'is_plugin': True,
            'command': method,
            'parameters': parameters,
            # 'uuid': self.uuid
        }
        run = Run(project=project, **record)
        # project.commit()
        return run

# def get_plugins_dir():
#     """
# Get the user directory where plugins are installed. The directory is created if it does not exist
# :return: the user plugin path
# :rtype: Path
# """
#     plugins_path = os.path.join(pydetecdiv.app.get_appdata_dir(), 'plugins')
#     if not os.path.exists(plugins_path):
#         os.mkdir(plugins_path)
#     return [plugins_path]
#     # return [pydetecdiv.plugins.__path__[0], plugins_path]


class PluginList:
    """
    Class to create and handle list of discovered plugins
    """

    def __init__(self):
        self.categories = []
        self.plugins_dict = {}

    @property
    def plugins(self):
        """
        return the mist of available plugins
        :return:
        """
        return self.plugins_dict.values()

    def load(self):
        """
        Discover plugins and load them in plugin list
        """
        for finder, name, _ in pkgutil.iter_modules(pydetecdiv.plugins.__path__):
            _ = self.load_plugin(finder, f'pydetecdiv.plugins.{name}')
        for finder, name, _ in pkgutil.iter_modules([get_plugins_dir()]):
            _ = self.load_plugin(finder, f'{name}')

    def load_plugin(self, finder, name):
        """
        Load a plugin given a finder and its module name

        :param finder: the module finder
        :param name: the module name
        :return: the plugin module
        """
        loader = finder.find_module(name)
        spec = importlib.util.spec_from_file_location(name, loader.path)
        module = importlib.util.module_from_spec(spec)
        # module = importlib.import_module(name)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if hasattr(module, 'Plugin'):
            if module.Plugin.category not in self.categories:
                self.categories.append(module.Plugin.category)
            self.plugins_dict[module.Plugin.id_] = module.Plugin()
            return module
        return None

    @property
    def len(self):
        """
        The number of discovered plugins

        :return: how many plugins have been discovered
        :rtype: int
        """
        return len(self.plugins)
