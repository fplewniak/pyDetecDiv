from pydetecdiv.domain.tools import Plugin

class Plugin_param(Plugin):
    def run(self):
        return {'stdout': self.parameters, 'stderr': ''}
