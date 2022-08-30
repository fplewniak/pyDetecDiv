"""
Definition of classes mapping exactly classes defined in the DetecDiv MATLAB version.
"""
import re
import json
import numpy as np
import jsonpickle


class Shallow:
    """
    The shallow object containing information about the DetecDiv poject
    """

    def __init__(self, tag: str = 'shallow project', io: object = None, fov=None, processing=None):
        """
        :param tag: a string
        :param io: IO object containing information about original data files
        :param fov: field of view in a FOV object
        :param processing: Processing object with information about processing of data
        """
        self.tag = tag
        self.io = IO() if io is None else IO(**io)
        self.fov = None if fov is None else FOV(**fov) if not isinstance(fov, list) else [FOV(**f) for f in fov]
        self.processing = None if processing is None else Processing(**processing)

    @property
    def json(self) -> str:
        """
        Converts the Shallow object into a string in JSON format.
        :return: a string in JSON format
        """
        json_str = jsonpickle.encode(self.__dict__)
        return re.sub('"py/object": "[^"]+", ', '', json_str)

    @classmethod
    def from_json(cls, json_str: str) -> object:
        """
        Creates a new Shallow object corresponding to a JSON string.
        :param json_str: a JSON string representing a Shallow object
        :return: the Shallow object created from the JSON string
        """
        json_dict = json.loads(json_str)
        return cls(**json_dict)

    @classmethod
    def read_json(cls, jsonfile: str) -> object:
        """
        Reads a JSON file containing a MATLAB DetecDiv shallow object and creates the corresponding Shallow object.
        :param jsonfile: the JSON file name
        :return: the Shallow object created from the JSON file
        """
        with open(jsonfile, 'r') as f:
            json_str = f.read()
        return cls.from_json(json_str)


class FOV:
    """
    Field Of View containing information about the complete field of view.
    """

    def __init__(self, id: str = '', srcpath: str = '', srclist: str = '', channel: list = None,
                 frames: list = None, interval: list = None, binning: list = None, contours: list = None,
                 tag: str = 'field of view', comments: str = '', number: int = 0, channels: int = 3,
                 orientation: int = 0, display: dict = None, crop: list = None, drift: list = None, roi=None,
                 flaggedROIs=None):
        self.id = id
        self.srcpath = srcpath
        self.srclist = srclist
        self.channel = [] if channel is None else channel
        self.frames = [] if frames is None else frames
        self.interval = [] if interval is None else interval
        self.binning = [] if binning is None else binning
        self.contours = [] if contours is None else contours
        self.tag = tag
        self.comments = comments
        self.number = number
        self.channels = channels
        self.orientation = orientation
        self.display = {} if display is None else display
        self.crop = [] if crop is None else crop
        self.drift = [] if drift is None else drift
        self.roi = None if roi is None else ROI(**roi) if not isinstance(roi, list) else [ROI(**r) for r in roi]
        self.flaggedROIs = None if flaggedROIs is None else ROI(**flaggedROIs) if not isinstance(flaggedROIs, list) \
            else [ROI(**r) for r in flaggedROIs]


class IO:
    """
    IO object containing the path to the original data.
    """

    def __init__(self, path: str = '', file: str = ''):
        self.path = path
        self.file = file


class ROI:
    """
    Region of Interest for analysis, i.e. a crop or the totalilty of the Field of view. Avoids using the whole field of
    view if unecessary.
    """

    def __init__(self, id: str = '', value: list = None, path: str = '', image: np.ndarray = None,
                 channelid: list = None, proc: list = None, parent: str = '', display: dict = None,
                 history: dict = None, classes: list = None, train: list = None, results: list = None, ):
        self.id = id
        self.value = [] if value is None else value
        self.path = path
        self.image = np.array([]) if image is None else image
        self.channelid = [] if channelid is None else channelid
        self.proc = [] if proc is None else proc
        self.parent = parent
        self.display = {} if display is None else display
        self.history = {} if history is None else history
        self.classes = [] if classes is None else classes
        self.train = [] if train is None else train
        self.results = [] if results is None else results  ## or a list of Display objects


class Processor:
    """
    Processor object providing information about data processor applied to regions of interest.
    """

    def __init__(self, id: int = 0, typeid: int = 0, path: str = '', strid: str = '', description: str = '',
                 category: str = '', param: list = None, processFun: str = '', processArg: dict = None,
                 history: list = None):
        self.id = id
        self.typeid = typeid
        self.path = path
        self.strid = strid
        self.description = description
        self.category = category
        self.param = [] if param is None else param
        self.processFun = processFun
        self.processArg = {} if processArg is None else processArg
        self.history = [] if history is None else history


class Processing:
    """
    Description of process applied to regions of interest
    """

    def __init__(self, roi: dict = None, classification: dict = None, processor: list = None):
        self.roi = {} if roi is None else roi
        self.classification = {} if classification is None else classification
        self.processor = None if processor is None else Processor(**processor) \
            if not isinstance(processor, list) else [Processor(**p) for p in processor]
