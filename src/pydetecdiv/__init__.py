'''
Utilities that can be useful in all places of the code
'''
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
    :param source:
    :param destination:
    """
    if platform.system() == 'Windows':
        cmd = ['copy']
    else:
        cmd = ['cp']
    return subprocess.Popen(cmd + list(source) + [destination])
    # os.popen(' '.join(cmd + list(source) + [destination]))

def delete_files(file_list):
    if platform.system() == 'Windows':
        cmd = ['del']
    else:
        cmd = ['rm', '-f']
    return subprocess.Popen(cmd + list(file_list))
    # os.popen(' '.join(cmd + list(file_list)))

