#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Settings management according to XDG base directory specification
http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
"""
import os
import configparser
from pathlib import Path
import xdg.BaseDirectory


def get_default_settings() -> dict:
    """
    Returns default values for configuration if no configuration file is found
    :return: a dictionary containing the default values
    """
    return {'project': {'dbms': 'SQLite3'},
            'project.sqlite': {'persistence': 'pydetecdiv.sqlite3', },
            'project.mysql': {'persistence': 'pydetecdiv', 'host': 'localhost', 'credentials': 'mysql.credentials', },
            'omero': {'host': 'localhost', 'credentials': 'omero.credentials', },
            }


def get_config_files():
    """
    Get a list configuration files conforming to the XDG Base directory specification for Linux and Mac OS or located
    in APPDATA folder for Microsoft Windows. This function does not check whether files exist as this is done anyway
    while trying to read configuration.
    :return: a list of configuration files
    """
    if 'APPDATA' in os.environ.keys():
        config_files = [Path(os.environ['APPDATA']).joinpath('pyDetecDiv').joinpath('settings.ini')]
    else:
        config_files = [Path(d).joinpath('settings.ini') for d in xdg.BaseDirectory.load_config_paths('pyDetecDiv')]
    return config_files


def get_config():
    """
    Get configuration parser from configuration files. If no file exists, then one is created in the favourite
    location with default values. Note that if the favourite directory does not exist either, it is created prior to
    saving the configuration.
    :return: a configuration parser
    """
    config_files = list(get_config_files())
    config = configparser.ConfigParser()
    config.read(config_files)
    if not config.sections():
        config.read_dict(get_default_settings())
        default_file = Path(xdg.BaseDirectory.save_config_path('pyDetecDiv')).joinpath('settings.ini')
        with open(default_file, 'w') as f:
            config.write(f)
    return config


def get_config_value(section: str, key: str):
    """
    Get value for a key in a section of the configuration file.
    :param section: the configuration section
    :param key: the configuration key
    :return: the corresponding value
    """
    config = get_config()
    return config.get(section, key)
