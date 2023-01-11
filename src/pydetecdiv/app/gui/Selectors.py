#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import json
import dearpygui.dearpygui as dpg
from pydetecdiv.app.gui import GenericWidget
from pydetecdiv.persistence.project import list_projects
from pydetecdiv.app.gui import object_pool


class GenericSelector(GenericWidget):
    def __init__(self, tag=0, label=None, combo_list=[], source=0, callback=None, **kwargs):
        super().__init__(tag, **kwargs)
        with dpg.group(horizontal=True, tag=tag, show=False):
            dpg.add_text(label)
            dpg.add_combo(combo_list, width=-1, tag=f'{tag}_combo', source=source, callback=callback)


class ProjectSelector(GenericSelector):
    def __init__(self, tag='project_selector', source=0, **kwargs):
        if tag != 'project_selector' and source == 0:
            source = 'project_selector'
        super().__init__(tag, 'Select project', list_projects(), source=f'{source}_combo', callback=self.select_project)
        self.show_hide()

    def select_project(self, sender, app_data, user_data):
        p = object_pool.close_project().set_project(app_data).project
        info_text = {
            'name': p.dbname,
            'author': p.author,
            'date': str(p.date),
        }
        dpg.set_value('info_text', json.dumps(info_text, indent=4))
        fov_list = [fov.name for fov in p.get_objects('FOV')]
        dpg.configure_item('fov_selector_combo', items=fov_list)
        dpg.set_value('fov_selector_combo', '')
        dpg.set_viewport_title(f'pyDetecDiv: {app_data}')


class FovSelector(GenericSelector):
    def __init__(self, tag='fov_selector', **kwargs):
        super().__init__(tag, 'Select FOV', [''], callback=self.select_fov)
        self.show_hide()

    def select_fov(self, sender, app_data, user_data):
        p = object_pool.project
        fov = p.get_named_object('FOV', app_data)
        data_files = p.get_linked_objects('Data', to=fov)
        d = f'{len(data_files)} files' if len(data_files) != 1 else data_files.name
        info_text = f"""
        Name: {fov.name}
        Size: {fov.size}
        Data files: {d}
        """
        dpg.set_value('info_text', info_text)
