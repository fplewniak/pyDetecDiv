'''
Utilities that can be useful in all places of the code
'''
import os
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
    :param source:
    :param destination:
    """
    if platform.system() == 'Windows':
        cp = 'copy'
    else:
        cp = 'cp'
    os.popen(f'{cp} {" ".join(source)} {destination}')
