#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import json
import dearpygui.dearpygui as dpg
from pydetecdiv.app.gui import object_pool

def select_project(sender, app_data, user_data):
    p = object_pool.set_project(app_data).project
    info_text = {
        'name': p.dbname,
        'author': p.author,
        'date': str(p.date),
                 }
    dpg.set_value('info_text', json.dumps(info_text, indent=4))
    fov_list = [fov.name for fov in p.get_objects('FOV')]
    dpg.configure_item('fov_selector_combo', items=fov_list)


def select_fov(sender, app_data, user_data):
    p = object_pool.project
    fov = p.get_named_object('FOV', app_data)
    data_files = p.get_linked_objects('Data', to=fov)
    d = f'{len(data_files)} files' if len(data_files) > 1 else data_files.name
    info_text = f"""
    Name: {fov.name}
    Size: {fov.size}
    Data files: {d}
    """
    dpg.set_value('info_text', info_text)

