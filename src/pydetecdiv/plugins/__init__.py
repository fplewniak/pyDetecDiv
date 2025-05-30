"""
Generic classes to discover and handle plugins
"""
import importlib
import pkgutil
import sys
from modulefinder import ModuleFinder
from typing import ValuesView, Self

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

import pydetecdiv
from pydetecdiv.domain.Project import Project
from pydetecdiv.plugins.parameters import Parameters
from pydetecdiv.settings import get_plugins_dir
from pydetecdiv.domain.Run import Run


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
        self.parameters = Parameters([])
        self.run = None

    def update_parameters(self, groups: list[str] | str = None) -> None:
        """
        Update the parameters in the provided list of group names

        :param groups: the list of group names or single group name
        """
        self.parameters.update(groups)
        self.parameters.reset(groups)

    def register(self) -> None:
        """
        Abstract method to register the plugin. This method should be implemented by all plugins for them to work.
        """
        raise NotImplementedError

    def addActions(self, menu: QMenu) -> None:
        """
        Method to add an action to a menu. This action triggers the launch method. If a submenu needs to be implemented
        or if arguments need to be passed to the launch method, this method may be overriden

        :param menu: the menu to add actions to
        """
        action = QAction(self.name, menu)
        action.triggered.connect(self.launch)
        menu.addAction(action)

    def launch(self) -> None:
        """
        Abstract method that needs to be implemented in each concrete Plugin implementation to launch the plugin (with
        or without a GUI)
        """
        raise NotImplementedError

    @property
    def parent_plugin(self) -> Self:
        """
        return the parent plugin of the current plugin. This may be used to add functionalities to plugins without
        having to modify the original code

        :return: None or the parent plugin
        """
        if self.parent in pydetecdiv.app.PyDetecDiv.plugin_list.plugins_dict:
            return pydetecdiv.app.PyDetecDiv.plugin_list.plugins_dict[self.parent]
        return None

    def save_run(self, project: Project, method: str, parameters: dict[str, object]) -> Run:
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
        self.run = Run(project=project, **record)
        # project.commit()
        return self.run


class PluginList:
    """
    Class to create and handle list of discovered plugins
    """

    def __init__(self):
        self.categories = []
        self.plugins_dict: dict[str, Plugin] = {}

    @property
    def plugins(self) -> ValuesView[Plugin]:
        """
        return the list of available plugins

        :return: the list of available plugins
        """
        return self.plugins_dict.values()

    def load(self) -> None:
        """
        Discover plugins and load them in plugin list
        """
        for finder, name, _ in pkgutil.iter_modules(pydetecdiv.plugins.__path__):
            _ = self.load_plugin(finder, f'pydetecdiv.plugins.{name}')
        for finder, name, _ in pkgutil.iter_modules([get_plugins_dir()]):
            _ = self.load_plugin(finder, f'{name}')

    def load_plugin(self, finder, name: str) -> Plugin | None:
        """
        Load a plugin given a finder and its module name

        :param finder: the module finder
        :param name: the module name
        :return: the plugin module
        """
        # loader = finder.find_module(name)
        # spec = importlib.util.spec_from_file_location(name, loader.path)
        spec = finder.find_spec(name)
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

    def register_all(self) -> None:
        """
        Register plugins
        """
        for plugin in self.plugins_dict.values():
            plugin.register()

    @property
    def len(self) -> int:
        """
        The number of discovered plugins

        :return: how many plugins have been discovered
        """
        return len(self.plugins)
