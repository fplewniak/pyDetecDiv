#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import json
import dearpygui.dearpygui as dpg
from pydetecdiv.app.gui import GenericWidget
from pydetecdiv.persistence.project import list_projects
from pydetecdiv.app.gui import register


class GenericSelector(GenericWidget):
    def __init__(self, tag=0, label=None, combo_list=None, source=0, callback=None, **kwargs):
        super().__init__(tag, **kwargs)
        self.combo = Combo(tag, label, combo_list, source, parent=self, **kwargs)

    def select(self, sender, app_data, user_data):
        ...

    def set(self, items, value):
        self.combo.set(items, value)

class Combo(GenericWidget):
    def __init__(self, tag, label, combo_list, source, parent, **kwargs):
        super().__init__(f'{tag}_combo', **kwargs)
        with dpg.group(horizontal=True, tag=tag, show=False):
            dpg.add_text(label)
            dpg.add_combo(combo_list, width=-1, tag=self.tag, source=source, callback=parent.select)

    def set(self, items, value):
        dpg.configure_item(self.tag, items=items)
        dpg.set_value(self.tag, value)

class ProjectSelector(GenericSelector):
    def __init__(self, tag='project_selector', source=0, **kwargs):
        if tag != 'project_selector' and source == 0:
            source = 'project_selector'
        super().__init__(tag, 'Select project', list_projects(), source=f'{source}_combo', callback=self.select)
        self.show_hide()

    def select(self, sender, app_data, user_data):
        register.close_project().set_project(app_data)
        dpg.set_value('info_text', json.dumps(register.project.record(), indent=4))
        register.get('fov_selector').set([fov.name for fov in register.project.get_objects('FOV')], '')
        register.get('roi_selector').set([], '')
        dpg.set_viewport_title(f'pyDetecDiv: {app_data}')


class FovSelector(GenericSelector):
    def __init__(self, tag='fov_selector', **kwargs):
        super().__init__(tag, 'Select FOV', [''], callback=self.select)
        self.show_hide()
        self.register()

    def select(self, sender, app_data, user_data):
        p = register.project
        fov = p.get_named_object('FOV', app_data)
        data_files = p.get_linked_objects('Data', to=fov)
        d = f'{len(data_files)} files' if len(data_files) != 1 else data_files.name
        info_text = f"""
        Name: {fov.name}
        Size: {fov.size}
        Data files: {d}
        ROI: {len(fov.roi_list)}
        """
        dpg.set_value('info_text', info_text)
        register.get('roi_selector').set([roi.name for roi in fov.roi_list], '')


class RoiSelector(GenericSelector):
    def __init__(self, tag='roi_selector', **kwargs):
        super().__init__(tag, 'Select ROI', [''], callback=self.select)
        self.show_hide()
        self.register()

    def select(self, sender, app_data, user_data):
        roi = register.project.get_named_object('ROI', app_data)
        data_files = register.project.get_linked_objects('Data', to=roi)
        d = f'{len(data_files)} files' if len(data_files) != 1 else data_files.name
        info_text = f"""
        Name: {roi.name}
        Position: {roi.top_left} - {roi.bottom_right}
        Data files: {d}
        """
        dpg.set_value('info_text', info_text)
