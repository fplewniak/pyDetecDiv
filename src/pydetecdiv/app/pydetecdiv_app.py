#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 The Graphical User Interface to pyDetecDiv application
"""
import importlib
import os

from PySide6.QtGui import QIcon

from pydetecdiv.app import create_app
from pydetecdiv.app.gui.Windows import MainWindow
import pydetecdiv.app.gui.SourcePath as SourcePath
from pydetecdiv.app.gui.core.parameters import StringParameter

if '_PYIBoot_SPLASH' in os.environ and importlib.util.find_spec("pyi_splash"):
    import pyi_splash

    pyi_splash.close()


def main_gui():
    """
    Main function for GUI application
    """
    app = create_app()

    style_sheet = """
                *:disabled {
                    color: gray; /* Set the text color to gray for disabled items */
                }
                MessageDialog QLabel {
                    font-weight: bold;
                 }
            """

    # Apply the style sheet to the application
    app.setStyleSheet(style_sheet)
    window_icon = QIcon(':icons/app_icon')
    app.setWindowIcon(window_icon)

    table_editor = SourcePath.TableEditor(title='Missing data source path definition')
    app.check_data_source_paths(table_editor)

    app.set_main_window(MainWindow())

    app.exec()


if __name__ == '__main__':
    main_gui()
