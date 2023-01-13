#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import numpy
import dearpygui.dearpygui as dpg

from pydetecdiv.app.gui import GenericWidget
import pydetecdiv.app.gui.Tools as Tools
import pydetecdiv.app.gui.Selectors as Selectors
from pydetecdiv.app.gui.Viewers import ImageViewer
from pydetecdiv.app.gui import registry


class GenericWindow(GenericWidget):
    def __init__(self, tag=0, label=None, **kwargs):
        super().__init__(tag, **kwargs)
        self.window = dpg.window(tag=tag, label=label, show=False)


class Explorer(GenericWindow):
    def __init__(self, **kwargs):
        super().__init__('explorer_window', 'Explorer')

        with self.window:
            dpg.add_text(tag='info_text', show=False)

            with dpg.collapsing_header(label='Projects', tag="Projects"):
                Selectors.ProjectSelector()

            with dpg.collapsing_header(label='Fields of View (FOVs)', tag="FOVs"):
                Selectors.FovSelector()

            with dpg.collapsing_header(label='Regions of Interest (ROIs)', tag="ROIs"):
                Selectors.RoiSelector()

            with dpg.collapsing_header(label='Datasets', tag="Datasets"):
                Selectors.DatasetSelector()

            with dpg.collapsing_header(label='Data files', tag="Data files"):
                data_selector = Selectors.DataSelector()
                dpg.add_button(label='View', callback=self.view_image, user_data=data_selector)

            with dpg.collapsing_header(label='History', tag="History"):
                dpg.add_text('Not implemented')

        self.show_hide()

    def view_image(self, sender, app_data, user_data):
        from pydetecdiv.domain.ImageResource import MemMapTiff
        registry.get('ImageViewer', default=ImageViewer()).clear().imshow(MemMapTiff(path=user_data.file_path, mode='r'))


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
            with dpg.group(horizontal=True):
                dpg.add_button(label='FOV', callback=self.view_image,
                               user_data='/data2/BioImageIT/workspace/fob1/Pos0_drift_corrected.tiff')
                dpg.add_button(label='ROI example', callback=self.view_image,
                               user_data='/data2/BioImageIT/workspace/fob1/ROI_example.tiff')
                dpg.add_button(label='same ROI with drift correction', callback=self.view_image,
                               user_data='/data2/BioImageIT/workspace/fob1/ROI_example_drift_corrected.tiff')
                dpg.add_button(label='2D image', callback=self.view_image,
                               user_data='/data2/BioImageIT/workspace/xxx/data/Pos9_5_89_frame_0572.tif')
        self.show_hide()

    def view_image(self, sender, app_data, user_data):
        from pydetecdiv.domain.ImageResource import MemMapTiff
        registry.get('ImageViewer', default=ImageViewer()).clear().imshow(MemMapTiff(path=user_data, mode='r'))


class Toolbox(GenericWindow):
    def __init__(self, **kwargs):
        super().__init__('tools_window', 'Toolbox')

        with self.window:
            Tools.CreateProject()
            Tools.ImportDataTool()
            Tools.ExtractROIs()
        self.show_hide()
