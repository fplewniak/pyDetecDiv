#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
from PySide6.QtCore import QSettings
from pydetecdiv.settings import get_config_files

project = None
main_window = None

def get_settings():
    return QSettings(str(get_config_files()[0]), QSettings.IniFormat)
