#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import dearpygui.dearpygui as dpg
from pydetecdiv.app.gui import GenericWidget
from pydetecdiv.app.gui import registry


class MenuBar(GenericWidget):
    def __init__(self, **kwargs):
        super().__init__('menu_bar', **kwargs)

        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Quit", callback=self.quit_application)
            with dpg.menu(label="Show/Hide"):
                dpg.add_menu_item(label="Explorer", callback=self.toggle_show, user_data='explorer_window')
                dpg.add_menu_item(label="Information", callback=self.toggle_show, user_data='info_window')
                dpg.add_menu_item(label="Viewer", callback=self.toggle_show, user_data='viewer_window')
                # dpg.add_menu_item(label="Workflow editor", callback=toggle_show, user_data='Workflow editor')
                dpg.add_menu_item(label="Toolbox", callback=self.toggle_show, user_data='tools_window')

    def quit_application(self, sender, app_data, user_data):
        registry.close_project()
        dpg.stop_dearpygui()

    def toggle_show(self, sender, app_data, user_data):
        dpg.configure_item(user_data, show=not dpg.get_item_configuration(user_data)['show'])
