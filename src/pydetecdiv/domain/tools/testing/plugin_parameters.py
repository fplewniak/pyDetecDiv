from pydetecdiv.domain.tools import Plugin

class Plugin_param(Plugin):
    def run(self):
        return {'stdout': self.parameters, 'stderr': ''}

class DSO_initializer(Plugin):
    def run(self):
        fov = self.parameters['fov']
        for roi in fov.roi_list:
            print(f'name: {roi.name}, top_left: {roi.top_left}, bottom_right: {roi.bottom_right}')
        return {'stdout':len(fov.roi_list), 'stderr': ''}
