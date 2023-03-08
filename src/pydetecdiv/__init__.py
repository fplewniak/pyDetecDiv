'''
Utilities that can be useful in all places of the code
'''
import os.path
import subprocess
import uuid
import platform


def generate_uuid():
    """
    Generate a universal id
    :return: uuid
    :rtype: str
    """
    return str(uuid.uuid4())


def copy_files(source, destination):
    """
    Copy files from source to destination with OS-specific command
    :param source: the list of files to copy
    :type source: list of str
    :param destination: the destination directory
    :type destination: str
    :return: the subprocess copying the files
    :rtype: subprocess.Popen
    """
    if platform.system() == 'Windows':
        destination = os.path.normpath(destination)
        cmd = []
        for path in [os.path.normpath(file_name) for file_name in list(source)]:
            cmd += ['copy', path, destination, '&']
        return subprocess.Popen(cmd, shell=True)
    cmd = ['cp'] + list(source) + [destination]
    return subprocess.Popen(cmd)


def delete_files(file_list):
    """
    Delete files with OS-specific command
    :param file_list: the list of files to delete
    :type file_list: list of str
    :return: the subprocess deleting the files
    :rtype: subprocess.Popen
    """
    if platform.system() == 'Windows':
        cmd = []
        for path in [os.path.normpath(file_name) for file_name in list(file_list)]:
            cmd += ['del', path, '&']
        return subprocess.Popen(cmd, shell=True)
    cmd = ['rm', '-f'] + list(file_list)
    return subprocess.Popen(cmd)
