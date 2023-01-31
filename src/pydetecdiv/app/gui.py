#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
 The Graphical User Interface to pyDetecDiv application
"""
from PySide6.QtWidgets import QApplication
import pydetecdiv.app.gui.Windows as Windows

if __name__ == '__main__':
    app = QApplication([])

    app.setApplicationName('pyDetecDiv')

    main_window = Windows.MainWindow()
    main_window.show()

    app.exec()

