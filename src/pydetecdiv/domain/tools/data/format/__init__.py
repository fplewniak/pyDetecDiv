from typing import Generator, Any

from pydetecdiv.app.parameters import Parameters, ChoiceParameter, DirParameter
from pydetecdiv.app.tools import Tool, Commands, Command
from pydetecdiv.domain.Project import Project


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
            Command('convert2ndtiff', 'Convert image files to NDTiff', self.convert_to_ndtiff)
            ])

        self.parameters.update_parameters(
                commands={'convert2ndtiff'},
                parameters=[
                    ChoiceParameter('paths', label=''),
                    ChoiceParameter('format', label='Format',
                                    items={
                                        'metadata'       : self.read_metadata,
                                        'Image directory': self.read_image_dir,
                                        }),
                    DirParameter('destination', 'Destination directory',)
                    ])

    def convert_to_ndtiff(self) -> Generator[float | int, Any, None]:
        """
        convert files to ndtiff
        """
        print('Counting data')
        file_count = 0
        for path, data_importer in self.parameters.paths.items:
            file_count += data_importer.count_data(path)
        print(f'Total files: {file_count}')

        for i in range(file_count):
            yield 100.0 * float(i + 1) / float(file_count)

    def read_metadata(self, filepath: str, project: Project):
        print(f'Read metadata: {filepath}')

    def read_image_dir(self, dirpath: str, project: Project):
        print(f'Read directory: {dirpath}')


