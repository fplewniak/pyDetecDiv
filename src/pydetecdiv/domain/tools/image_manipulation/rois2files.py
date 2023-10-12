"""
Plugins for testing parameters
"""
import os
from datetime import datetime

from aicsimageio import AICSImage

from pydetecdiv import generate_uuid
from pydetecdiv.domain.tools import Plugin
from pydetecdiv.settings import get_config_value


class SaveROIsToFiles(Plugin):
    """
    Test initialization of DSOs
    """
    def run(self):
        """
        run the plugin
        :return: output
        """
        project = self.parameters['fov'].dso[0].project
        roi_list = []
        for fov in self.parameters['fov'].dso:
            for roi in fov.roi_list:
                roi_list.append(roi)
                x_slice = slice(roi.top_left[0], roi.bottom_right[0])
                y_slice = slice(roi.top_left[1], roi.bottom_right[1])
                roi_image = AICSImage(fov.image_resource().image_resource_data().data_sample(X=x_slice, Y=y_slice))
                print(f'{roi_image} {roi_image.shape}')
                file_name = os.path.join(self.working_dir, f'{roi.name}.tiff')
                dataset = project.get_named_object('Dataset', self.dataset)
                record = {
                    'id_': None,
                    'uuid': generate_uuid(),
                    'name': f'{roi.name}.tiff',
                    'dataset': dataset.uuid,
                    'author': get_config_value('project', 'user'),
                    'date': datetime.now(),
                    'url': f'{roi.name}.tiff',
                    'format': self.parameters['roifiles'].format,
                    'source_dir': '',
                    'meta_data': '{}',
                    'key_val': '{}',
                }
                data = project.get_object('Data', project.save_record('Data', record))
                project.link_objects(data, roi)
                roi_image.save(file_name)
                print(file_name)
                # TODO if do not apply drift then return slice directly, otherwise, determine slice one image after the
                # TODO other, applying drift correction each time and stack all images. This will also allow to create
                # TODO subsets of videos, using only some channels, layers and for a shorter period of time
                # TODO those informations (T, C, Z span should appear in the run options and maybe the file names)
        return {'stdout': [r.name for r in roi_list], 'stderr': ''}
