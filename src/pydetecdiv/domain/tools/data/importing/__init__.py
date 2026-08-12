"""
Tools to import data into a project
"""
import glob
import os
from typing import Generator, Any, cast

import polars

from pydetecdiv import utils
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, set_connections

from pydetecdiv.app.parameters import Parameters, ChoiceParameter, FileParameter
from pydetecdiv.app.tools import Tool, Commands, Command
from pydetecdiv.domain.Classification import Classification
from pydetecdiv.domain.FOV import FOV
from pydetecdiv.domain.Project import Project
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.domain.RoiAnnotations import RoiAnnotations


class DataImportTool(Tool):
    id_ = 'cnrs.plewniak.dataimport'
    version = '1.0.0'
    name = 'Import data'

    def __init__(self, parameters: Parameters = Parameters(), commands: Commands = Commands(), working_dir: str | None = None):
        super().__init__(parameters=parameters, commands=commands, working_dir=working_dir)

        self.commands.update([
            Command('import_images', 'Image files', self.import_files),
            Command('import_annotated_rois', 'Annotated ROIs', self.import_annotated_rois),
            Command('create_resources', 'Create Image resources', self.create_image_resources),
            ])

        self.parameters.update_parameters(
                commands={'import_images'},
                parameters=[
                    ChoiceParameter('paths', label='', all_values=True),
                    ChoiceParameter('format', label='Format',
                                    items={
                                        'metadata'       : self.import_metadata,
                                        'NDTiff'         : self.import_ndtiff,
                                        'Image directory': self.import_image_dir,
                                        }),
                    ])

        self.parameters.update_parameters(
                commands={'import_annotated_rois'},
                parameters=[
                    FileParameter('roi_annotation_file', 'ROI annotation file', require_existing=True),
                    ChoiceParameter('classification', label='Class names', updater=self.update_classification)
                    ])

        set_connections({PyDetecDiv.app.project_selected: [self.update_classification]})

    @staticmethod
    def import_metadata(filepath: str, project: Project) -> Generator[int, int, None]:
        """
        Import image files using MicroManager metadata files
        :param filepath: the path to the metadata file(s)
        :param project: the project
        """
        metadata_file_names = [f for f in glob.glob(filepath) if os.path.isfile(f)]
        for metadata_file_name in metadata_file_names:
            for i in project.import_images_from_metadata(metadata_file_name):
                yield i

    @staticmethod
    def import_image_dir(dirpath: str, project: Project) -> Generator[int, int, None]:
        """
        Import image files from directories
        :param dirpath: the path to the directories
        :param project: the project
        """
        image_dirs = [f for f in glob.glob(dirpath) if os.path.isdir(f) and utils.check_contains_tiff(f)]
        for image_dir in image_dirs:
            for i in project.import_images_in_dir(image_dir):
                yield i

    @staticmethod
    def import_ndtiff(dirpath: str, project: Project) -> Generator[int, int, None]:
        """
        Import image NDTiff datasets
        :param dirpath: the path to the datasets
        :param project: the project
        """
        ndtiff_dirs = [f for f in glob.glob(dirpath) if os.path.isdir(f) and utils.check_is_ndtiff(f)]
        for ndtiff_dir in ndtiff_dirs:
            for i in project.import_ndtiff_data(ndtiff_dir):
                yield i

    def update_classification(self) -> None:
        """
        Update the list of classification schemes available in the current project
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            cast(ChoiceParameter, self.parameters.classification).set_items(
                    {f'{c.name} {c.classes}': c for c in cast(list[Classification], project.get_objects('Classification'))})

    def import_files(self) -> Generator[float | int, Any, None]:
        """
        Import files
        """
        print('Counting data')
        file_count = 0
        for path, data_importer in self.parameters.paths.items:
            file_count += data_importer.count_data(path)
        print(f'Total files: {file_count}')

        run = self.save_run()

        if file_count:
            with pydetecdiv_project(PyDetecDiv.project_name) as project:
                count = 0
                for path, data_importer in self.parameters.paths.items:
                    for i in data_importer.import_func(path, project):
                        yield 100 * float(count + i) / float(file_count)
                    count += i
                project.commit()
        self.parameters.paths.clear()

    def import_annotated_rois(self) -> Generator[float | int, Any, None]:
        """
        Import ROIs and annotations from a csv file
        """
        print('Import annotated ROIs from file')
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            fov_list = {cast(FOV, fov).name: fov.id_ for fov in project.get_objects('FOV')}
            roi_names = [cast(ROI, roi).name for roi in project.get_objects('ROI')]
            annotated_rois = polars.read_csv(self.parameters.roi_annotation_file.value).filter(polars.col('fov').is_in(fov_list))
            class_names = set(annotated_rois.select('class_name').unique('class_name').to_numpy().flatten())

            classification = self.parameters.classification.value

            if not set(class_names).issubset(set(classification.classes)):
                print('Invalid class names: not compatible with the current classification scheme.')
                return
            run = self.save_run()
            i = 0
            for row in annotated_rois.iter_rows(named=True):
                if row['roi'] not in roi_names:
                    new_roi = ROI(project=project, name=row['roi'], fov=fov_list[row['fov']],
                                  top_left=(row['x'], row['y']),
                                  bottom_right=(row['x'] + row['width'], row['y'] + row['height']))
                    roi_names.append(new_roi.name)
                _ = RoiAnnotations(project=project, roi=new_roi.id_, t=row['frame'],
                                   classification=classification, annotation=row['class_name'],
                                   run=run, key_val={'class_name': row['class_name']})
                i += 1
                yield 100 * float(i) / float(len(annotated_rois))
            project.commit()

    def create_image_resources(self) -> Generator[float | int, Any, None]:
        ...

