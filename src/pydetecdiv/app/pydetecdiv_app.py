#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 The Graphical User Interface to pyDetecDiv application
"""
from PySide6.QtGui import QIcon
import pyqtgraph as pg
from pydetecdiv.app.gui.tools import ToolAction
from pydetecdiv.app.gui.tools.data.format import DataFormatMenu

from pydetecdiv.app.gui.tools.data.hdf5 import Create_ROI_HDF5Dialog

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui import FileMenu, ProjectMenu, DataMenu, Enable #, PluginMenu
from pydetecdiv.app.gui.Windows import MainWindow
from pydetecdiv.app.gui import SourcePath
from pydetecdiv.app.gui.tools.data.importing import DataImportMenu
from pydetecdiv.app.gui.tools.deep_learning.classification_schemes import ClassificationSchemeMenu
from pydetecdiv.app.gui.tools.deep_learning.model_info import ModelInfoMenu
from pydetecdiv.app.gui.tools.video_classifier import VideoClassifierMenu
from pydetecdiv.app.gui.RawData2FOV import RawData2FOV
from pydetecdiv.domain.tools.data.format import DataFormat
from pydetecdiv.domain.tools.data.hdf5 import ROIseqHDF5creator
from pydetecdiv.domain.tools.data.importing import DataImportTool
from pydetecdiv.domain.tools.deep_learning.classification_schemes import ClassificationSchemeManagement
from pydetecdiv.domain.tools.deep_learning.model_info import ModelInfo
from pydetecdiv.domain.tools.video_classifier import VideoClassifier


# if '_PYIBoot_SPLASH' in os.environ and importlib.util.find_spec("pyi_splash"):
#     import pyi_splash
#
#     pyi_splash.close()


def main_gui():
    """
    Main function for GUI application
    """
    PyDetecDiv.app = PyDetecDiv([])
    pg.setConfigOptions(antialias=True, background='w')

    style_sheet = """
                * {
                    font-family: Arial;
                }
                *:disabled {
                    color: gray; /* Set the text color to gray for disabled items */
                }
                QDialog QLabel {
                    font-family: Arial;
                    font-size: 10pt;
                 }
                 QTabWidget::red {
                    font-size: 20pt;
                 }
                 QHeaderView {
                    font-style: italic;
                    font-weight: bold;
                    color: white;
                    background: #bbb;
                 }
            """

    # Apply the style sheet to the application
    PyDetecDiv.app.setStyleSheet(style_sheet)
    window_icon = QIcon(':icons/app_icon')
    PyDetecDiv.app.setWindowIcon(window_icon)

    # Check table sources for the current machine
    table_editor = SourcePath.TableEditor(title='Missing data source path definition', editable_col=None)
    PyDetecDiv.check_data_source_paths(table_editor)

    # Set main application window
    mw = PyDetecDiv.set_main_window(MainWindow())

    # Create tools
    PyDetecDiv.update_tools({'cnrs.plewniak.videoclassifier'      : VideoClassifier(working_dir='video_classification'),
                             'cnrs.plewniak.roiseqhdf5creator'    : ROIseqHDF5creator(working_dir='data'),
                             'cnrs.plewniak.classificationschemes': ClassificationSchemeManagement(working_dir='data'),
                             'cnrs.plewniak.deeplearningmodelinfo': ModelInfo(working_dir='data'),
                             'cnrs.plewniak.dataimport'           : DataImportTool(working_dir='data'),
                             'cnrs.plewniak.dataformat'           : DataFormat(working_dir='data'),
                             })

    video_tools = [
        VideoClassifierMenu('cnrs.plewniak.videoclassifier'),
        ]

    deeplearning_tools = [
        ClassificationSchemeMenu('cnrs.plewniak.classificationschemes'),
        ModelInfoMenu('cnrs.plewniak.deeplearningmodelinfo'),
        ]

    data_tools = [
        DataImportMenu('cnrs.plewniak.dataimport', enable=Enable.if_project_exists),
        ToolAction('cnrs.plewniak.dataimport', 'create_resources', RawData2FOV,
                   enable=Enable.if_missing_image_resources),
        None,
        ToolAction('cnrs.plewniak.roiseqhdf5creator', 'create_roi_hdf5',
                   Create_ROI_HDF5Dialog,  enable=Enable.if_rois),
        DataFormatMenu('cnrs.plewniak.dataformat', enable=Enable.if_project_exists)
        ]

    # Create menus
    FileMenu(mw)
    ProjectMenu(mw)
    DataMenu(mw)
    mw.add_top_menus({
        'Data'         : data_tools,
        'Deep learning': deeplearning_tools,
        'Video'        : video_tools,
        })

    # Launch application GUI
    PyDetecDiv.app.exec()


if __name__ == '__main__':
    main_gui()
