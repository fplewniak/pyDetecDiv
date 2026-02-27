#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 The Graphical User Interface to pyDetecDiv application
"""
# import importlib
# import os

from PySide6.QtGui import QIcon
import pyqtgraph as pg

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui import FileMenu, ProjectMenu, DataMenu, PluginMenu, VideoMenu
from pydetecdiv.app.gui.Windows import MainWindow
from pydetecdiv.app.gui import SourcePath
from pydetecdiv.app.gui.tools.video_classifier import VideoClassifierMenu
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
    PyDetecDiv.plugin_list.register_all()
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
            """

    # Apply the style sheet to the application
    PyDetecDiv.app.setStyleSheet(style_sheet)
    window_icon = QIcon(':icons/app_icon')
    PyDetecDiv.app.setWindowIcon(window_icon)

    # Check table sources for the current machine
    table_editor = SourcePath.TableEditor(title='Missing data source path definition', editable_col=None)
    PyDetecDiv.app.check_data_source_paths(table_editor)

    # Create tools
    PyDetecDiv.app.update_tools({'Video classifier': VideoClassifier(working_dir='video_classifier')})
    video_tool_menus = [
        VideoClassifierMenu(PyDetecDiv.app.tools['Video classifier']),
        ]

    # Set main application window
    PyDetecDiv.app.set_main_window(MainWindow())

    # Create menus
    FileMenu(PyDetecDiv.main_window)
    ProjectMenu(PyDetecDiv.main_window)
    DataMenu(PyDetecDiv.main_window)
    VideoMenu(PyDetecDiv.main_window, video_tool_menus)
    PluginMenu(PyDetecDiv.main_window)

    # Launch application GUI
    PyDetecDiv.app.exec()


if __name__ == '__main__':
    main_gui()
