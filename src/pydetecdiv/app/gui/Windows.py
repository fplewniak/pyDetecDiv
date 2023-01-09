#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import dearpygui.dearpygui as dpg
from pydetecdiv.app.gui import GenericWidget
import pydetecdiv.app.gui.Tools as Tools
import pydetecdiv.app.gui.Selectors as Selectors


class GenericWindow(GenericWidget):
    def __init__(self, tag, label, **kwargs):
        super().__init__(tag, **kwargs)
        self.window = dpg.window(tag=tag, label=label, show=False)


class Explorer(GenericWindow):
    def __init__(self, **kwargs):
        super().__init__('explorer_window', 'Explorer')

        with self.window:
            dpg.add_text(tag='info_text', show=False)
            with dpg.collapsing_header(label='Projects', tag="Projects"):
                Selectors.ProjectSelector()
            with dpg.collapsing_header(label='Datasets', tag="Datasets"):
               ...
            with dpg.collapsing_header(label='Data files', tag="Data files"):
                ...
            with dpg.collapsing_header(label='Fields of View (FOVs)', tag="FOVs"):
                Selectors.FovSelector()
            with dpg.collapsing_header(label='Regions of Interest (ROIs)', tag="ROIs"):
                ...
            with dpg.collapsing_header(label='History', tag="History"):
                ...

        self.show_hide()


class Information(GenericWindow):
    def __init__(self, **kwargs):
        super().__init__('info_window', 'Information')

        with self.window:
            dpg.add_text(source='info_text')
        self.show_hide()


class Viewer(GenericWindow):
    def __init__(self, **kwargs):
        super().__init__('viewer_window', 'Viewer')

        with self.window:
            ...
        self.show_hide()


class Toolbox(GenericWindow):
    def __init__(self, **kwargs):
        super().__init__('tools_window', 'Toolbox')

        with self.window:
            Tools.ImportDataTool()
            Tools.ExtractROIs()
        self.show_hide()
