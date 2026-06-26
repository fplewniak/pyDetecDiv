"""
Tools to import data into a project
"""
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

    def import_files(self):
        for path, import_func in self.parameters.paths.items:
            import_func(path)

    def import_metadata(self, path):
        print(path, 'import metadata')

    def import_image_files(self, path):
        print(path, 'import image files')

    def import_image_dir(self, path):
        print(path, 'import image directory')

    def import_ndtiff(self, path):
        print(path, 'import ndtiff')
