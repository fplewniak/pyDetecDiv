"""
Tools for management of classification schemes used by deep learning classifiers
"""
import polars

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.app.tools import Tool, Commands, Command

from pydetecdiv.app.parameters import Parameters, StringParameter, StringListParameter
from pydetecdiv.domain.Classification import Classification


class ClassificationSchemeManagement(Tool):
    """
    Classification scheme management tool
    """
    id_ = 'cnrs.plewniak.classificationschemes'
    version = '1.0.0'
    name = 'Classification schemes'

    def __init__(self, parameters: Parameters = Parameters(), commands: Commands = Commands(), working_dir: str | None = None):
        super().__init__(parameters=parameters, commands=commands, working_dir=working_dir)

        self.commands.update([
            Command('manage_schemes', 'Manage classification schemes', self.delete_removed_schemes)
            ])

        self.parameters.update_parameters([
            StringParameter('name', label='Name'),
            StringListParameter('classes', label='Classes'),
            ])

    def save_scheme(self):
        """
        Saves the scheme whose name is in self.parameters.name parameter
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            scheme = project.get_named_object('Classification', self.parameters.name.value)
            if scheme is None:
                scheme = Classification(project=project, name=self.parameters.name.value, classes=self.parameters.classes.value,
                                        key_val={})
            else:
                scheme.classes = self.parameters.classes.value
                scheme.validate(updated=True)

    def delete_removed_schemes(self, data: polars.DataFrame) -> None:
        """
        Deletes classification schemes records in repository that were removed from the table view.
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            for row in project.get_polars('Classification').join(data,
                                                                 left_on='name', right_on='name', how='anti').iter_rows(named=True):
                project.delete(project.get_object('Classification', row['id_']))

    @staticmethod
    def scheme_is_not_used(scheme_name: str):
        """
        Tests whether the scheme is used in the project

        :param scheme_name: the scheme name
        :return: True if scheme is NOT used, False otherwise
        """
        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            scheme = project.get_named_object('Classification', scheme_name)
            return not project.has_links('RoiAnnotations', scheme)
