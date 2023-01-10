#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import dearpygui.dearpygui as dpg
from pydetecdiv.app.gui import GenericWidget
from pydetecdiv.app.gui.callbacks import select_project, select_fov
from pydetecdiv.persistence.project import list_projects


class GenericSelector(GenericWidget):
    def __init__(self, tag, label, combo_list, source=0, callback=None, **kwargs):
        super().__init__(tag, **kwargs)
        with dpg.group(horizontal=True, tag=tag, show=False):
            dpg.add_text(label)
            dpg.add_combo(combo_list, width=-1, tag=f'{tag}_combo', source=source, callback=callback)


class ProjectSelector(GenericSelector):
    def __init__(self, tag='project_selector', source=0, **kwargs):
        if tag != 'project_selector' and source == 0:
            source = 'project_selector'
        super().__init__(tag, 'Select project', list_projects(), source=f'{source}_combo', callback=select_project)
        self.show_hide()


class FovSelector(GenericSelector):
    def __init__(self, tag='fov_selector', **kwargs):
        super().__init__(tag, 'Select FOV', [''], callback=select_fov)
        self.show_hide()
