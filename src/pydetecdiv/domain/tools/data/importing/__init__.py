"""
Tools to import data into a project
"""
import glob
import os
from typing import Generator

from pydetecdiv import utils

from pydetecdiv.app.parameters import Parameters, ChoiceParameter
from pydetecdiv.app.tools import Tool
from pydetecdiv.domain.Project import Project


class DataImportTool(Tool):
    id_ = 'cnrs.plewniak.dataimport'
    version = '1.0.0'
    name = 'Import data'

    def __init__(self, parameters: Parameters = Parameters(), working_dir: str | None = None):
        super().__init__(parameters=parameters, working_dir=working_dir)

        self.parameters.add_parameters(
                [
                    ChoiceParameter('paths', label=''),
                    ChoiceParameter('format', label='Format',
                                    items={
                                        'metadata'       : self.import_metadata,
                                        'NDTiff'         : self.import_ndtiff,
                                        'Image directory': self.import_image_dir,
                                        }),
                    ]
                )

    def import_metadata(self, filepath: str, project: Project) -> Generator[int, int, None]:
        """
        Import image files using MicroManager metadata files
        :param filepath: the path to the metadata file(s)
        :param project: the project
        """
        metadata_file_names = [f for f in glob.glob(filepath) if os.path.isfile(f)]
        for metadata_file_name in metadata_file_names:
            for i in project.import_images_from_metadata(metadata_file_name):
                yield i

    # def import_image_files(self, path, project) -> Generator[int, int, None]:
    #     print(path, 'import image files')
    #     yield 1
    #
    # def count_image_files(self, path) -> int:
    #     print('counting image files')
    #     return 1

    def import_image_dir(self, dirpath: str, project: Project) -> Generator[int, int, None]:
        """
        Import image files from directories
        :param dirpath: the path to the directories
        :param project: the project
        """
        image_dirs = [f for f in glob.glob(dirpath) if os.path.isdir(f) and utils.check_contains_tiff(f)]
        for image_dir in image_dirs:
            for i in project.import_images_in_dir(image_dir):
                yield i

    def import_ndtiff(self, dirpath: str, project: Project) -> Generator[int, int, None]:
        """
        Import image NDTiff datasets
        :param dirpath: the path to the datasets
        :param project: the project
        """
        ndtiff_dirs = [f for f in glob.glob(dirpath) if os.path.isdir(f) and utils.check_is_ndtiff(f)]
        for ndtiff_dir in ndtiff_dirs:
            for i in project.import_ndtiff_data(ndtiff_dir):
                yield i
