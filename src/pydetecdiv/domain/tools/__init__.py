import importlib
import inspect


class Plugins:
    def __init__(self, path):
        self.list = {name: cls for name, cls in inspect.getmembers(importlib.import_module(path))}


class Plugin:
    def __init__(self, parameters):
        self.parameters = parameters

    def run(self):
        ...
