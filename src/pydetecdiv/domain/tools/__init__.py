"""
Tool plugins
"""
import importlib
import inspect


class Plugins:
    """
    The list of plugins available in a module defined by its path
    """
    def __init__(self, path):
        self.list = dict(inspect.getmembers(importlib.import_module(path)))


class Plugin:
    """
    Generic plugin class that must be inherited by concrete plugins implementing the run() method
    """
    def __init__(self, parameters):
        self.parameters = parameters

    def run(self):
        """
        Method implementing the actual algorithm for the plugin
        """
        raise NotImplementedError
