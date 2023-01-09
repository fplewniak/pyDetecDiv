#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import dearpygui.dearpygui as dpg
from pydetecdiv.app.gui import GenericWidget
import pydetecdiv.app.gui.Selectors as Selectors

class GenericTool(GenericWidget):
    def __init__(self, tag, label, **kwargs):
        super().__init__(tag, **kwargs)
        self.tool = dpg.collapsing_header(label=label, tag=tag, show=False)


class ImportDataTool(GenericTool):
    def __init__(self, **kwargs):
        super().__init__('importdata_tool', 'Import Data')
        with self.tool:
            Selectors.ProjectSelector(tag='import_project_selector')
        self.show_hide()

class ExtractROIs(GenericTool):
    def __init__(self, **kwargs):
        super().__init__('extract_rois_tool', 'Extract ROIs')
        with self.tool:
            pass
            # ProjectSelector()

        self.show_hide()
