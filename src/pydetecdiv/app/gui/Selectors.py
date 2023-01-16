#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import abc
import dearpygui.dearpygui as dpg
from pydetecdiv.app.gui import GenericWidget
from pydetecdiv.persistence.project import list_projects
from pydetecdiv.app.gui import registry


class GenericSelector(GenericWidget):
    def __init__(self, tag=0, label=None, combo_list=None, source=0, **kwargs):
        super().__init__(tag, **kwargs)
        self.dso = None
        with dpg.group(horizontal=True, tag=tag, show=False):
            dpg.add_text(label)
            dpg.add_combo(combo_list, width=-1, tag=f'{self.tag}_combo', source=source, callback=self.select)

    @abc.abstractmethod
    def select(self, sender, app_data, user_data):
        """
        Abstract method enforcing the implementation of the select callback in all selectors
        """

    def set(self, items, value):
        dpg.configure_item(f'{self.tag}_combo', items=items)
        dpg.set_value(f'{self.tag}_combo', value)

    @property
    def value(self):
        dpg.get_value(f'{self.tag}_combo')


class ProjectSelector(GenericSelector):
    def __init__(self, tag='project_selector', source=0, **kwargs):
        if tag != 'project_selector' and source == 0:
            source = 'project_selector'
        super().__init__(tag, 'Select project', list_projects(), source=f'{source}_combo', **kwargs)
        self.show_hide()

    def select(self, sender, app_data, user_data):
        self.dso = registry.close_project().set_project(app_data).project
        registry.get('fov_selector').set([fov.name for fov in registry.project.get_objects('FOV')], '')
        registry.get('roi_selector').set([], '')
        registry.get('dataset_selector').set([dataset.name for dataset in registry.project.get_objects('Dataset')], '')
        registry.get('data_selector').set([], '')
        dpg.set_value('info_text', registry.project.info)
        dpg.set_viewport_title(f'pyDetecDiv: {app_data}')


class FovSelector(GenericSelector):
    def __init__(self, tag='fov_selector', **kwargs):
        super().__init__(tag, 'Select FOV', [''], **kwargs)
        self.register().show_hide()

    def select(self, sender, app_data, user_data):
        self.dso = registry.project.get_named_object('FOV', app_data)
        registry.get('roi_selector').set([roi.name for roi in self.dso.roi_list], '')
        registry.get('data_selector').set([data.name for data in self.dso.data], '')
        dpg.set_value('info_text', self.dso.info)


class RoiSelector(GenericSelector):
    def __init__(self, tag='roi_selector', **kwargs):
        super().__init__(tag, 'Select ROI', [''], **kwargs)
        self.register().show_hide()

    def select(self, sender, app_data, user_data):
        self.dso = registry.project.get_named_object('ROI', app_data)
        dpg.set_value('info_text', self.dso.info)


class DataSelector(GenericSelector):
    def __init__(self, tag='data_selector', **kwargs):
        super().__init__(tag, 'Select Data', [''], **kwargs)
        self.register().show_hide()

    def select(self, sender, app_data, user_data):
        self.dso = registry.project.get_named_object('Data', app_data)
        dpg.set_value('info_text', self.dso.info)

    @property
    def file_path(self):
        return self.dso.url


class DatasetSelector(GenericSelector):
    def __init__(self, tag='dataset_selector', **kwargs):
        super().__init__(tag, 'Select Dataset', [''], **kwargs)
        self.register().show_hide()

    def select(self, sender, app_data, user_data):
        self.dso = registry.project.get_named_object('Dataset', app_data)
        registry.get('data_selector').set([data.name for data in self.dso.data_list], '')
        dpg.set_value('info_text', self.dso.info)
