"""
Plugins for testing parameters
"""
from pydetecdiv.domain.tools import Plugin


class PluginParam(Plugin):
    """
    Test of parameters as passed to plugin
    """
    def run(self):
        """
        run the plugin
        :return: output
        """
        for _, p in self.parameters.items():
            print(f'{p.name}: {p.value} {p.obj}')
        return {'stdout': self.parameters, 'stderr': ''}


class DSOinitializer(Plugin):
    """
    Test initialization of DSOs
    """
    def run(self):
        """
        run the plugin
        :return: output
        """
        fov = self.parameters['fov'].obj
        for roi in fov.roi_list:
            print(f'name: {roi.name}, top_left: {roi.top_left}, bottom_right: {roi.bottom_right}')
        return {'stdout': len(fov.roi_list), 'stderr': ''}
