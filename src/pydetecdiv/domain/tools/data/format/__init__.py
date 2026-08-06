"""
Data format handling and conversion
"""
import glob
import json
import os
from typing import Generator, Any

import tifffile
from ndtiff import NDTiffDataset

from pydetecdiv import utils

from pydetecdiv.app.parameters import Parameters, ChoiceParameter, DirParameter
from pydetecdiv.app.tools import Tool, Commands, Command


class DataFormat(Tool):
    """
    Class defining the tool managing and converting data formats
    """
    id_ = 'cnrs.plewniak.dataformat'
    version = '1.0.0'
    name = 'Data format tool'

    def __init__(self, parameters: Parameters = Parameters(), commands: Commands = Commands(), working_dir: str | None = None):
        super().__init__(parameters, commands, working_dir)

        self.commands.update([
            Command('metadata2ndtiff', 'Convert image files to NDTiff', self.metadata_to_ndtiff)
            ])

        self.parameters.update_parameters(
                commands={'metadata2ndtiff'},
                parameters=[
                    ChoiceParameter('paths', label=''),
                    DirParameter('destination', 'Destination directory', default='./NDTiff')
                    ])

    def metadata_to_ndtiff(self) -> Generator[float | int, Any, None]:
        """
        Wrapping the conversion method, sending progress information as percentage
        """
        print('Counting data')
        file_count = 0
        for path, _ in self.parameters.paths.items:
            file_count += utils.count_metadata(path)
        print(f'Total files: {file_count}')

        for i in self.convert_metadata():
            yield 100.0 * float(i) / float(file_count)

    def convert_metadata(self) -> Generator[float | int, Any, Any]:
        """
        Convert files listed in MicroManager metadata files to ndtiff
        """
        metadata_file_names = [f for path in self.parameters.paths.keys for f in glob.glob(path) if os.path.isfile(f)]

        with open(metadata_file_names[0]) as f:
            summary_metadata = json.load(f)['Summary']
        if summary_metadata['Width'] == 0:
            summary_metadata['Width'] = -1
        if summary_metadata['Height'] == 0:
            summary_metadata['Height'] = -1
        dataset = NDTiffDataset(self.parameters.destination.value, summary_metadata=summary_metadata, writable=True)

        count = 0

        for path in self.parameters.paths.keys:
            print(f'Read metadata: {path}')
            for metadata_file_name in metadata_file_names:
                with open(metadata_file_name) as metadata_file:
                    metadata = json.load(metadata_file)
                    summary = metadata['Summary']
                    for d in [v for k, v in metadata.items() if k.startswith('Metadata-')]:
                        image_coordinates = {'channel' : d['ChannelIndex'], 'time': d['FrameIndex'], 'z': d['SliceIndex'],
                                             'position': d['PositionIndex']
                                             }
                        pixels = tifffile.imread(os.path.join(os.path.dirname(metadata_file_name), os.path.basename(d["FileName"])))
                        d['PositionName'] = summary['StagePositions'][d['PositionIndex']]['Label']
                        dataset.put_image(image_coordinates, pixels, d)
                        count += 1
                        yield count
