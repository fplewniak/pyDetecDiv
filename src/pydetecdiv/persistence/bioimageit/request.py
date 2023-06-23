# -*- coding: utf-8 -*-
"""Main REST like API extension of the BioImageIT original to add pyDetecDiv-specific methods

This file implements the main BioImageIT API with using stateless operation like for a REST API
"""

import os.path

import bioimageit_core.api.request
from bioimageit_core import ConfigAccess
from bioimageit_core.core.exceptions import DataServiceError
from bioimageit_core.core.utils import format_date


class Request(bioimageit_core.api.request.Request):
    """
    A class extending the BioIMageIT Request class in order to add some features without having to modify the original
    code.
    """
    def create_experiment(self, name, author='', date='now', keys=None, destination=''):
        """Create a new experiment

        **Parameters**
        name: str
            Name of the experiment
        author: str
            username of the experiment author
        date: str
            Creation date of the experiment
        keys: list
            List of keys used for the experiment vocabulary
        destination: str
            Destination where the experiment is created. It is a the path of the
            directory where the experiment will be created for local use case
        **Returns**
        Experiment container with the experiment metadata

        """
        if keys is None:
            keys = []
        try:
            if author == '':
                author = ConfigAccess.instance().config['user']['name']
            return self.data_service.create_experiment(name, author, format_date(date),
                                                       keys, destination)
        except DataServiceError as err:
            self.notify_error(str(err))

    def annotate_raw_data_using_regex(self, experiment, keys_, regex):
        """
        Annotate raw dataset of an experiment using the source path name

        :param experiment: the experiment
        :type experiment: Experiment Container
        :param keys_: the annotation keys
        :type keys_: tuple of str
        :param regex: the regular expression defining how to extract the values corresponding to the specified keys
        :type regex: regular expression str
        """
        print(f'Please, wait while annotating raw dataset of {experiment.name}')
        self.data_service.annotate_using_regex(experiment, experiment.raw_dataset,
                                               lambda x: os.path.join(x['source_dir'], x['name']), keys_, regex)
        print('OK')

    def annotate_using_regex(self, experiment, dataset, source, keys_, regex):
        """
        Annotate a dataset of an experiment using a regular expression

        :param experiment: the experiment containing the dataset to annotate
        :type experiment: Experiment Container
        :param dataset: the dataset to annotate
        :type dataset: Dataset Container or DatasetInfo Container
        :param source: a string (column name) or callable returning a string that the regular expression will be
        applied to
        :type source: callable or str
        :param keys_: the annotation keys
        :type keys_: tuple of str
        :param regex: the regular expression defining how to extract the values corresponding to the specified keys
        :type regex: regular expression str
        """
        print(f'Please, wait while annotating dataset {dataset.name}')
        self.data_service.annotate_using_regex(experiment, dataset, source, keys_, regex)
        print('OK')

    def clear_annotation(self, experiment, dataset, key):
        """
        Clear annotations with the specified key in a dataset

        :param experiment: the experiment containing the dataset to update
        :type experiment: Experiment Container
        :param dataset: the dataset
        :type dataset: Dataset Container or DatasetInfo Container
        :param key: the key to remove from the dataset annotations
        :type key: str
        """
        self.data_service.clear_annotation(experiment, dataset, key)

    def import_glob(self, experiment, files_glob, author='',
                    format_='imagetiff', date='now', destination=None, **kwargs):
        """
        Import into an experiment a list of raw data files specified by a glob pattern

        :param experiment: the experiment
        :type experiment: Experiment Container
        :param files_glob: the file path specification
        :type files_glob: str
        :param author: the author's name
        :type author: str
        :param format_: the file format
        :type format_: str
        :param date: the date
        :type date: str
        :param observers: a list of observers to notify progress
        :type observers: list of Observer objects
        """
        # print(f'Please wait while importing data into {experiment.raw_dataset.name} dataset')
        self.data_service.import_glob(experiment, files_glob,
                                      author=author, format_=format_, date=date, observers=None,
                                      destination=destination)
        # print('OK')

    def run(self, job):
        """Run a BioImageIT job

        A BioImageIT job is a run of a processing tool in a database. The data to process are
        selected in the database using a specified request, and the results are automatically
        saved in a new dataset of the database. All the metadata of the job (tool, request,
        parameters) are also saved in the database.
        """
        if job.tool.type == "merge":
            self._run_job_merged(job)
        else:
            self._run_job_sequence(job)
        return job.experiment
