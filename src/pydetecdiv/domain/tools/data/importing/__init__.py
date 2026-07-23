"""
Tools to import data into a project
"""
import glob
import json
import os
from pathlib import Path
from typing import AnyStr

import polars
from ndtiff import NDTiffDataset

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
            for i in project.import_images_from_metadata(metadata_file_name):
                yield i

    def count_metadata(self, path):
        metadata_file_names = [f for f in glob.glob(path) if os.path.isfile(f)]
        file_count = 0
        for metadata_file_name in metadata_file_names:
            with open(metadata_file_name) as metadata_file:
                metadata = json.load(metadata_file)
                file_count += len([v for k, v in metadata.items() if k.startswith('Metadata-')])
        print(f'counting metadata: {file_count} image files')
        return file_count

    def import_image_files(self, path, project):
        print(path, 'import image files')

    def count_image_files(self, path):
        print('counting image files')

    def import_image_dir(self, path, project):
        print(path, 'import image directory')

    def count_image_dir(self, path):
        print('counting image files')

    def import_ndtiff(self, path, project):
        ndtiff_dirs = [f for f in glob.glob(path) if os.path.isdir(f) and self.check_is_ndtiff(f)]
        for ndtiff_dir in ndtiff_dirs:
            for i in project.import_ndtiff_data(ndtiff_dir):
                yield i

    def count_ndtiff(self, path):
        ndtiff_dirs = [f for f in glob.glob(path) if os.path.isdir(f) and self.check_is_ndtiff(f)]
        ndtiff_dir_count = 0
        for ndtiff_dir in ndtiff_dirs:
            ndtiff_ds = NDTiffDataset(str(ndtiff_dir))
            df = polars.DataFrame(ndtiff_ds.get_image_coordinates_list())
            dims_df = df.group_by(by='position').agg(polars.col('time').max(), polars.col('z').max(), polars.col('channel').max())
            ndtiff_dir_count += dims_df.select(polars.len()).item()

        print(f'counting ndtiff: {ndtiff_dir_count} NDTIff FOV dataset')
        return ndtiff_dir_count

    @staticmethod
    def check_is_ndtiff(directory: str | bytes) -> bool:
        """
        Check whether the specified path is a NDTiff path and enables the Ok button if it is
        """
        return directory != '' and os.path.isfile(os.path.join(str(directory), 'NDTiff.index'))
