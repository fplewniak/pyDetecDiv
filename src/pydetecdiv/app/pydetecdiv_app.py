#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 The Graphical User Interface to pyDetecDiv application
"""
from PySide6.QtGui import QIcon

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.Windows import MainWindow


def main_gui():
    """
    Main function for GUI application
    """
    app = PyDetecDiv([])

    window_icon = QIcon(':icons/app_icon')
    app.setWindowIcon(window_icon)
    app.main_window = MainWindow()
    app.main_window.show()

    app.exec()


if __name__ == '__main__':
    main_gui()
