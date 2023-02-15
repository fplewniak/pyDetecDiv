#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 The Graphical User Interface to pyDetecDiv application
"""
from PySide6.QtGui import QIcon

from pydetecdiv.app import PyDetecDivApplication
from pydetecdiv.app.gui.Windows import MainWindow

if __name__ == '__main__':
    app = PyDetecDivApplication([])

    window_icon = QIcon(':icons/app_icon')
    app.setWindowIcon(window_icon)
    PyDetecDivApplication.main_window = MainWindow()
    PyDetecDivApplication.main_window.show()

    app.exec()
