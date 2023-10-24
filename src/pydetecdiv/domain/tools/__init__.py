"""
Tool plugins
This code is deprecated and will be replaced with code in pydetecdiv.plugins module
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
    def __init__(self, parameters, dataset, working_dir):
        self.parameters = parameters
        self.dataset = dataset
        self.working_dir = working_dir

    def run(self):
        """
        Method implementing the actual algorithm for the plugin
        """
        raise NotImplementedError
