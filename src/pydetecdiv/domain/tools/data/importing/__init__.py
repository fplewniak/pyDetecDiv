"""
Tools to import data into a project
"""
import glob
import json
import os

from pydetecdiv.app.parameters import Parameters, ChoiceParameter
from pydetecdiv.app.tools import Tool


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

    def import_metadata(self, path, project):
        metadata_file_names = [f for f in glob.glob(path) if os.path.isfile(f)]
        for metadata_file_name in metadata_file_names:
                project.import_images_from_metadata(metadata_file_name)

    def count_metadata(self, path):
        metadata_file_names = [f for f in glob.glob(path) if os.path.isfile(f)]
        file_count = 0
        for metadata_file_name in metadata_file_names:
            with open(metadata_file_name) as metadata_file:
                metadata = json.load(metadata_file)
                file_count += len([v for k, v in metadata.items() if k.startswith('Metadata-')])
        print(f'counting metadata: {file_count} image files')
        return file_count

    def import_image_files(self, path):
        print(path, 'import image files')

    def count_image_files(self, path):
        print('counting image files')

    def import_image_dir(self, path):
        print(path, 'import image directory')

    def count_image_dir(self, path):
        print('counting image files')

    def import_ndtiff(self, path):
        print(path, 'import ndtiff')

    def count_ndtiff(self, path):
        print('counting ndtiff')
