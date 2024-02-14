#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 The Graphical User Interface to pyDetecDiv application
"""
import importlib
import os

from PySide6.QtGui import QIcon

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.Windows import MainWindow

if '_PYIBoot_SPLASH' in os.environ and importlib.util.find_spec("pyi_splash"):
    import pyi_splash
    pyi_splash.close()

def main_gui():
    """
    Main function for GUI application
    """
    app = PyDetecDiv([])

    style_sheet = """
                *:disabled {
                    color: gray; /* Set the text color to gray for disabled items */
                }
            """

    # Apply the style sheet to the application
    app.setStyleSheet(style_sheet)

    window_icon = QIcon(':icons/app_icon')
    app.setWindowIcon(window_icon)
    app.main_window = MainWindow()
    app.main_window.show()

    app.exec()


if __name__ == '__main__':
    main_gui()
