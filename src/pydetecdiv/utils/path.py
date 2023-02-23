#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
A utility module for path manipulation
"""
import os


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]
