"""
Plugins for testing parameters
"""
from pydetecdiv.domain.tools import Plugin


class SaveROIsToFiles(Plugin):
    """
    Test initialization of DSOs
    """
    def run(self):
        """
        run the plugin
        :return: output
        """
        roi_list = []
        for fov in self.parameters['fov'].dso:
            for roi in fov.roi_list:
                roi_list.append(roi)
                # print(f'name: {roi.name}, top_left: {roi.top_left}, bottom_right: {roi.bottom_right}')
        return {'stdout': len(roi_list), 'stderr': ''}
