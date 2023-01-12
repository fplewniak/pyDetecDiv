#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import dearpygui.dearpygui as dpg
from pydetecdiv.app.gui import GenericWidget
import pydetecdiv.app.gui.Selectors as Selectors
from pydetecdiv.app.gui import registry
from pydetecdiv.persistence.project import list_projects


class GenericTool(GenericWidget):
    def __init__(self, tag, label, **kwargs):
        super().__init__(tag, **kwargs)
        self.tool = dpg.collapsing_header(label=label, tag=tag, show=False)


class CreateProject(GenericTool):
    def __init__(self, **kwargs):
        super().__init__('createproject_tool', 'Create Project')
        with self.tool:
            with dpg.group(horizontal=True):
                dpg.add_text('Project name:')
                dpg.add_input_text(width=-1, callback=self.create_project, on_enter=True)
        self.show_hide()

    def create_project(self, sender, app_data, user_data):
        registry.set_project(app_data)
        dpg.configure_item('project_selector_combo', items=list_projects())


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
