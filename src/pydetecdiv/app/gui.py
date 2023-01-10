#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 The Graphical User Interface to pyDetecDiv application
"""
import dearpygui.dearpygui as dpg
from pathlib import Path
from pydetecdiv.settings import get_config_dir
import pydetecdiv.app.gui.Windows as Windows
from pydetecdiv.app.gui.MenuBar import MenuBar

if __name__ == '__main__':
    dpg.create_context()

    dpg.configure_app(docking=True, docking_space=True, init_file=Path(get_config_dir()).joinpath('gui.ini'))
    dpg.create_viewport(title=f'pyDetecDiv', width=800, height=600, clear_color=[0, 0, 255, 255])
    dpg.setup_dearpygui()

    menu = MenuBar()

    explorer = Windows.Explorer()
    info = Windows.Information()
    toolbox = Windows.Toolbox()
    viewer = Windows.Viewer()

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
