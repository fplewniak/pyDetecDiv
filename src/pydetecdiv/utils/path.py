#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
A utility module for path manipulation
"""
import os
import glob


def stem(path: str) -> str:
    """
    Return the basename of a file without extension

    :param path: the file path
    :type path: str
    :return: the file basename without extension
    :rtype: str
    """
    return os.path.splitext(os.path.basename(path))[0]

def files_in_dir(directory: str, extensions: list[str]) -> list[str]:
    file_list = []
    for extension in extensions:
        file_list.extend(glob.glob(os.path.join(directory, extension)))
    return file_list
