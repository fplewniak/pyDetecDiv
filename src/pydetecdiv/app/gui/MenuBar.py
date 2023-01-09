#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import dearpygui.dearpygui as dpg
from pydetecdiv.app.gui import GenericWidget

def toggle_show(sender, app_data, user_data):
    dpg.configure_item(user_data, show=not dpg.get_item_configuration(user_data)['show'])

class MenuBar(GenericWidget):
    def __init__(self, **kwargs):
        super().__init__('menu_bar')

        with dpg.viewport_menu_bar():
            with dpg.menu(label="Show/Hide"):
                dpg.add_menu_item(label="Explorer", callback=toggle_show, user_data='explorer_window')
                dpg.add_menu_item(label="Information", callback=toggle_show, user_data='info_window')
                dpg.add_menu_item(label="Viewer", callback=toggle_show, user_data='viewer_window')
                # dpg.add_menu_item(label="Workflow editor", callback=toggle_show, user_data='Workflow editor')
                dpg.add_menu_item(label="Toolbox", callback=toggle_show, user_data='tools_window')
