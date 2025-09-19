#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Settings management according to XDG base directory specification
http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
"""
import datetime
import os
import getpass
import configparser
import platform
from uuid import UUID, uuid5, SafeUUID, getnode
from pathlib import Path

import polars
import xdg.BaseDirectory
from polars import read_csv, col

from pydetecdiv.utils import increment_string


class UUID_NameSpace:
    """
    Name space definition for the creation of uuid5 unique ids
    """
    DataSource = '29082025-9dad-11d1-80b4-00c04fd430c8'


def get_default_settings() -> dict:
    """
    Returns default values for configuration if no configuration file is found

    :return: a dictionary containing the default values
    """
    return {'project'       : {'dbms' : 'SQLite3', 'workspace': get_default_workspace_dir(),
                               'user' : getpass.getuser(),
                               'batch': 1024
                               },
            'project.sqlite': {'database': 'pydetecdiv'},
            'project.conda' : {'dir': '/opt/miniconda3/'},
            'paths'         : {'appdata'        : xdg.BaseDirectory.save_data_path('pyDetecDiv'),
                               'sam2_checkpoint': '/data/SegmentAnything2/checkpoints/sam2.1_hiera_large.pt',
                               'sam2_model_cfg' : 'configs/sam2.1/sam2.1_hiera_l.yaml',
                               'toolbox'        : '/data/BioImageIT/bioimageit-toolboxes'
                               }
            # 'project.mysql': {'database': 'pydetecdiv', 'host': 'localhost', 'credentials': 'mysql.credentials', },
            # 'omero': {'host': 'localhost', 'credentials': 'omero.credentials', },
            # 'bioimageit': {'config_file': '/data2/BioImageIT/config.json'}
            }


def get_config_dir() -> Path:
    """
    Returns the directory containing the pyDetecDiv configuration files

    :return: the config directory path
    """
    if 'APPDATA' in os.environ:
        config_dir = Path(os.environ['APPDATA']).joinpath('pyDetecDiv')
    else:
        config_dir = xdg.BaseDirectory.load_first_config('pyDetecDiv')
    return config_dir


def get_config_files() -> list[Path]:
    """
    Get a list configuration files conforming to the XDG Base directory specification for Linux and Mac OS or located
    in APPDATA folder for Microsoft Windows. This function does not check whether files exist as this is done anyway
    while trying to read configuration.

    :return: a list of configuration files
    """
    if 'APPDATA' in os.environ:
        config_files = [Path(os.environ['APPDATA']).joinpath('pyDetecDiv').joinpath('settings.ini')]
    else:
        config_files = [Path(d).joinpath('settings.ini') for d in xdg.BaseDirectory.load_config_paths('pyDetecDiv')]
    return config_files


def get_config_file() -> Path:
    """
    Gets the first configuration file in the list of available configuration files

    :return: the path f the configuration file
    """
    return get_config_files()[0]


def get_config() -> configparser.ConfigParser:
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


def get_config_value(section: str, key: str) -> str:
    """
    Get value for a key in a section of the configuration file.

    :param section: the configuration section
    :param key: the configuration key
    :return: the corresponding value
    """
    config = get_config()
    return config.get(section, key)


def get_appdata_dir() -> Path:
    """
    get the local Application directory (.local/share/pyDetecDiv on Linux, AppData\pyDetecDiv on Windows)

    :return: the path to the local application directory
    """
    if not get_config().has_option('paths', 'appdata'):
        return xdg.BaseDirectory.save_data_path('pyDetecDiv')
    return Path(get_config_value('paths', 'appdata'))


def get_plugins_dir() -> Path:
    """
    Get the user directory where plugins are installed. The directory is created if it does not exist

    :return: the user plugin path
    """
    plugins_path = Path(os.path.join(get_appdata_dir(), 'plugins'))
    if not os.path.exists(plugins_path):
        os.mkdir(plugins_path)
    return plugins_path


def get_default_workspace_dir() -> Path:
    """
    Get the user workspace directory. The default directory is not created if it does not exist to avoid confusion. It is up to the
    user to make sure the directory exists or select an existing directory.

    :return: the user workspace path
    """
    default_workspace_dir = Path(os.path.join('/data', getpass.getuser(), 'workspace'))
    # if not os.path.exists(default_workspace_dir):
    #     os.mkdir(default_workspace_dir)
    return default_workspace_dir


def datapath_file(datapath_filename: str = '.datapath_list.csv') -> Path:
    """
    Return the path to data source list file

    :return: the path to the data source list file
    """
    return Path(os.path.join(get_config_value('project', 'workspace'), datapath_filename))


def datapath_list(datapath_filename: str = '.datapath_list.csv', grouped: bool = False) -> polars.DataFrame | polars.dataframe.group_by.GroupBy:
    """
    Returns a DataFrame containing the data source dir definitions

    :return: data source dir definitions
    """
    datapath_list_file = datapath_file(datapath_filename)
    if not datapath_list_file.is_file():
        polars.DataFrame({'name'   : [],
                          'path_id': [],
                          'device' : [],
                          'MAC'    : [],
                          'path'   : []
                          }).write_csv(datapath_list_file)
    datapath_list = read_csv(datapath_list_file)
    if grouped:
        return datapath_list.group_by(by='path_id')
    return datapath_list


def create_path_id(as_string: bool = False) -> UUID | str:
    """
    Creates a unique identifier for data source path, from the device name and current time

    :return: the uuid5
    """
    uuid = uuid5(UUID(UUID_NameSpace.DataSource, is_safe=SafeUUID.safe), Device.name() + str(datetime.datetime.now()))
    if as_string:
        return str(uuid)
    return uuid


def all_path_ids(df: polars.DataFrame) -> polars.DataFrame:
    """
    Returns all defined path ids on all devices

    :param df: the DataFrame containing the data source path definitions
    :return: a DataFrame containing only path ids
    """
    return df.select(col('path_id')).unique()


class Device:
    """
    A class handling device-specific methods, used to get information about the current device and adapt configuration thereto
    """

    @classmethod
    def name(cls) -> str:
        """
        returns the name of the current device

        :return: the device name
        """
        return platform.node()

    @classmethod
    def mac(cls) -> str:
        """
        returns the MAC address of the current device

        :return: the device MAC address
        """
        return ":".join(("%012X" % getnode())[i: i + 2] for i in range(0, 12, 2))

    @classmethod
    def add_path(cls, name: str, path: Path | str, path_id: str = None) -> None:
        """
        Adds a new path specification for the current device. If path_id is None (i.e. this data source has not been set already on any
        device) then a new id is generated from the MAC address and the current time. Otherwise, the specified path id is used, which is
        the case to add a path specification for a path that was created on another device.
        If the pair of values (path, MAC address) is already present in the list, then the name value of the corresponding row is
        updated.

        :param name: the user-defined name of the data source on the current device
        :param path: the path of the data source on the current device
        :param path_id: the cross-device path id of the data source
        """
        df = datapath_list()
        if path_id is None:
            path_id = create_path_id(as_string=True)
        new_row = polars.DataFrame({'name'   : [name],
                                    'path_id': [path_id],
                                    'device' : [Device.name()],
                                    'MAC'    : [Device.mac()],
                                    'path'   : [path]
                                    })
        mask_mac_path = (df['MAC'] == cls.mac()) & (df['path'] == path)

        if mask_mac_path.any():
            df = df.with_columns(
                    polars.when(mask_mac_path).then(polars.lit(name)).otherwise(df['name']).alias('name')
                    )
        else:
            df = df.extend(new_row)
        df.write_csv(datapath_file())

    @classmethod
    def data_path(cls, path_id: str) -> str | None:
        """
        Returns the data source path corresponding to the path_id on the current device

        :param path_id: the path id
        :return: the data source path corresponding to the path_id on the current device
        """
        df = datapath_list()
        path = (df.filter((col('path_id') == path_id)
                          & ((col('MAC') == cls.mac()) | (col('device') == cls.name()))
                          ))
        if path.shape[0] > 0:
            return path.select(col('path')).item()
        return None

    @classmethod
    def path_id(cls, path: str) -> str | None:
        """
        Given a path, finds the corresponding root path (mounting point) for the current device and returns the path variable.

        :param path: the path
        :return: the path variable id to use in the database
        """
        df = datapath_list()
        paths = (df.filter((col('MAC') == cls.mac()) | (col('device') == cls.name()))).select(col('path'), col('path_id'))
        for p, v in paths.rows():
            if p == os.path.commonpath([p, path]):
                return v
        return None

    @classmethod
    def path_ids(cls, df: polars.DataFrame) -> polars.DataFrame:
        """
        Returns path ids defined on the current device

        :param df: the dataframe containing the list of data source path definitions
        :return: a dataframe with the path ids that are defined for the current device
        """
        return (df.filter((col('MAC') == cls.mac()) | (col('device') == cls.name()))).select(col('path_id')).unique()

    @classmethod
    def undefined_path_ids(cls) -> polars.DataFrame:
        """
        Returns path ids undefined on the current device

        :return: a dataframe of path ids that are not defined on the current device
        """
        df = datapath_list()
        return all_path_ids(df).join(cls.path_ids(df), on=['path_id'], how='anti')

    @classmethod
    def undefined_paths(cls) -> polars.DataFrame:
        """
        Returns all data source paths defined on other devices that are not defined on the current one.

        :return: a dataframe with the data source paths
        """
        return datapath_list().join(cls.datapath_list(), on=['path_id'], how='anti')

    @classmethod
    def datapath_list(cls) -> polars.DataFrame:
        """
        Returns a list of data source paths defined on the current device, identified by the MAC address or the device name.

        :return: a dataframe with the list of data source paths
        """
        return datapath_list().filter((col('MAC') == cls.mac()) | (col('device') == cls.name()))


    @classmethod
    def get_path_id_and_url(cls, abs_url) -> polars.DataFrame:
        """
        Returns the path id and relative url corresponding to the absolute url on the current device .

        :return: a tuple containing the path_id and the relative url
        """
        path_id = cls.path_id(abs_url)
        if path_id is not None:
            url = os.path.relpath(abs_url, start=Device.data_path(path_id))
        else:
            path_id = os.path.dirname(abs_url)
        return path_id, url
