import importlib
import pkgutil

import pydetecdiv


class Plugin:
    id = None
    version = None
    name = None
    category = None

    def __init__(self, **kwargs):
        ...

    # def register_object(self, obj):
    #     ...


class PluginList:
    def __init__(self):
        self.categories = []
        self.plugins = []

    def load(self):
        for _, name, _ in pkgutil.iter_modules(pydetecdiv.plugins.__path__):
            module = importlib.import_module(f'pydetecdiv.plugins.{name}')
            if module.Plugin.category not in self.categories:
                self.categories.append(module.Plugin.category)
            self.plugins.append(module.Plugin())
        print(self.categories)
        for plugin in self.plugins:
            print(f'{plugin.name} ({plugin.category}):')
            print(f'   id: {plugin.id} version: {plugin.version}')

    @property
    def len(self):
        return len(self.plugins)
