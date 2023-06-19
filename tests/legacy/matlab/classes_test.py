from pydetecdiv.legacy.matlab.classes import Shallow, IO
from deepdiff import DeepDiff
import jsonpickle
import hypothesis.strategies as st
from hypothesis import given, settings
import glob


def get_files(files: str) -> list:
    """
    Returns a list of files corresponding to a glob pattern.
    :param files: the glob pattern for file selection
    :return: a list of files defined by the glob pattern in files argument
    """
    return [f for f in glob.glob(files)]


@given(json_file=st.sampled_from(get_files('data/legacy/classes_test/*.json')))
@settings(deadline=None)
def shallow_json_file_test(json_file):
    """
    Tests whether Shallow objects correctly read from json files.
    :param json_file: the glob pattern for json file selection
    """
    shallow = jsonpickle.decode(Shallow.read_json(json_file).json)
    with open(json_file, 'r') as f:
        json_dict = jsonpickle.decode(f.read())
        assert DeepDiff(json_dict, shallow) == {}
