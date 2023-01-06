# -*- coding: utf-8 -*-
"""SQLite3 metadata service.

This module implements the SQLite3 service for metadata
(Data, DataSet and Experiment) management.
This service read/write and query metadata from a SQLite3 database

Classes
-------
SQLiteMetadataServiceBuilder
SQLiteMetadataService

"""
import glob
import platform
import os
import os.path
from datetime import datetime
import json
import re
from shutil import copyfile
import subprocess
import pandas
import zarr
from skimage.io import imread

from bioimageit_formats import FormatsAccess, formatsServices
from bioimageit_core.core.config import ConfigAccess
from bioimageit_core.core.utils import generate_uuid, format_date
from bioimageit_core.core.exceptions import DataServiceError
from bioimageit_core.containers.data_containers import (METADATA_TYPE_RAW,
                                                        METADATA_TYPE_PROCESSED,
                                                        Container,
                                                        RawData,
                                                        ProcessedData,
                                                        Dataset,
                                                        Experiment,
                                                        Run,
                                                        DatasetInfo,
                                                        )
from bioimageit_core.plugins.data_local import LocalMetadataService
import sqlalchemy


class SQLiteMetadataServiceBuilder:
    """Service builder for the metadata service"""

    def __init__(self):
        self._instance = None
        # self.db = None

    def __call__(self, **_ignored):
        if not self._instance:
            self._instance = SQLiteMetadataService()
        return self._instance


class SQLiteMetadataService(LocalMetadataService):
    """Service for SQLite3-based metadata management"""

    def __init__(self):
        self.service_name = 'SQLiteMetadataService'
        self.repository = None
        self.session = None

    def connect_to_session(self, session, repository):
        """
        Associate an open SQLite session to the data service
        :param session: A SQLite session
        :type session: sqlalchemy.orm.session.Session
        """
        self.session = session
        self.repository = repository

    def determine_links_using_regex(self, dataset, source, keys_, regex):
        """
        Determines links between data in dataset and other objects in database. The returned DataFrame can be used to
        populate a table in SQLite database
        :param dataset: the dataset
        :type dataset: Dataset Container or DatasetInfo Container
        :param source: a string (column name) or callable returning a string that the regular expression will be
        applied to
        :type source: callable or str
        :param keys_: the column names (one object class per column)
        :type keys_: tuple of str
        :param regex: the regular expression defining how to extract the values corresponding to the specified keys
        :type regex: regular expression str
        :return: a DataFrame representing the links between data and objects (one object per column)
        :rtype: pandas DataFrame
        """
        con = self.session.connection()
        df = pandas.read_sql_query('SELECT * from data', con)

        pattern = re.compile(regex)

        call_back = source if callable(source) else lambda x: x[source]

        links = list()
        for i in df.index[df['dataset'] == dataset.uuid].to_list():
            m = re.search(pattern, call_back(df.loc[i,]))
            if m:
                key_val = {'id_': df.loc[i, 'id_']}
                key_val.update(dict(zip(keys_, m.groups())))
                links.append(key_val)
        return pandas.DataFrame(links)

    def clear_annotation(self, dataset, key):
        """
        Clear annotations with the specified key in a dataset
        :param experiment: the experiment containing the dataset to update
        :type experiment: Experiment Container
        :param dataset: the dataset
        :type dataset: Dataset Container or DatasetInfo Container
        :param key: the key to remove from the dataset annotations
        :type key: str
        """
        self.session.execute(sqlalchemy.text(
            f'UPDATE data SET key_val = json_remove(key_val, "$.{key}") WHERE dataset = "{dataset.uuid}";'))

    def create_annotations_using_regex(self, dataset, source, keys_, regex):
        """
        Use regular expression to create a DataFrame containing annotations of data in a dataset. Basic usage is as
        follows:
        create_annotations_using_regex(experiment, experiment.raw_dataset,
          lambda x: os.path.join(x['source_dir'],x['name']) ,
          ('group', 'ROI', 'FOV', 'position', 'frame'),
          r'.*\/([^\/]+)\/(Pos(\d+)_(\d+_\d+))_frame_(\d\d\d\d).tif')
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
        :return: a DataFrame representing the data table with updated annotations
        :rtype: pandas DataFrame
        """
        con = self.session.connection()
        df = pandas.read_sql_query('SELECT * from data', con)

        pattern = re.compile(regex)

        call_back = source if callable(source) else lambda x: x[source]

        for i in df.index[df['dataset'] == dataset.uuid].to_list():
            m = re.search(pattern, call_back(df.loc[i,]))
            if m:
                key_val = json.loads(df.loc[i, 'key_val'])
                key_val.update(dict(zip(keys_, m.groups())))
                df.loc[i, 'key_val'] = json.dumps(key_val)
        return df

    def annotate_using_regex(self, experiment, dataset, source, keys_, regex):
        """
        Use DataFrame returned by create_annotations_using_regex method to annotate data in dataset
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
        df = self.create_annotations_using_regex(experiment, dataset, source, keys_, regex)
        con = self.session.connection()
        con.execute('DELETE from data')
        df.to_sql('data', con, if_exists='append', index=False)
        experiment.keys = [key[0] for key
                           in con.execute('SELECT DISTINCT key FROM json_each(key_val), data').fetchall()]

    def create_experiment(self, name, author='', date='now', keys=None, destination=''):
        """Create a new experiment

        Parameters
        ----------
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

        Returns
        -------
        Experiment container with the experiment metadata

        """
        if keys is None:
            keys = []
        container = Experiment()
        container.id_ = None
        container.uuid = generate_uuid()
        container.name = name
        container.author = author
        container.date = date
        container.keys = keys

        # check the destination dir
        if destination == '':
            destination = ConfigAccess.instance().config['workspace']
        uri = os.path.abspath(destination)
        if not os.path.exists(uri):
            raise DataServiceError(
                'Cannot create Experiment: the destination '
                'directory does not exists'
            )

        uri = os.path.abspath(uri)

        # create the experiment directory
        filtered_name = name.replace(' ', '')
        experiment_path = os.path.join(uri, filtered_name)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)
        else:
            raise DataServiceError(
                'Cannot create Experiment: the experiment '
                'directory already exists'
            )

        # create an empty raw dataset
        raw_dataset = Dataset()
        raw_dataset.id_ = None
        raw_dataset.uuid = generate_uuid()
        raw_dataset.url = experiment_path
        raw_dataset.md_uri = (experiment_path, raw_dataset.id_)
        raw_dataset.name = 'data'
        os.mkdir(os.path.join(experiment_path, raw_dataset.name))
        raw_dataset.type = METADATA_TYPE_RAW
        raw_dataset.format = None
        raw_dataset.run = None

        raw_dataset.md_uri = self.update_dataset(raw_dataset)
        container.raw_dataset = DatasetInfo(raw_dataset.name, raw_dataset.md_uri,
                                            raw_dataset.uuid)
        container.raw_dataset.id_ = raw_dataset.md_uri

        # save the experiment metadata in SQLite3 database
        self.update_experiment(container)
        return container

    def get_workspace_experiments(self, workspace_uri):
        """Read the experiments in the user workspace

        Parameters
        ----------
        workspace_uri: str
            URI of the workspace

        Returns
        -------
        list of experiment containers
        """
        if os.path.exists(workspace_uri):
            dirs = os.listdir(workspace_uri)
            experiments = []
            for dir_ in dirs:
                exp_path = os.path.join(workspace_uri, dir_, f'{dir_}.db')
                if os.path.exists(exp_path):
                    experiments.append({'md_uri': exp_path, 'info': self.get_experiment(exp_path)})
            return experiments
        return []

    def get_experiment(self, md_uri):
        """Read an experiment from the database

        Parameters
        ----------
        md_uri: str
            URI of the experiment. For local use case, the URI is either the
            path of the experiment directory, or the path of the
            experiment.md.json file

        Returns
        -------
        Experiment container with the experiment metadata

        """
        #md_uri = os.path.abspath(md_uri)
        experiment_path = os.path.join(ConfigAccess.instance().config['workspace'], md_uri)
        if os.path.isdir(experiment_path):
            container = Experiment()
            container.md_uri = md_uri
            container.id_, container.uuid, container.name, container.author, container.date, rds = self.session.execute(
                'SELECT * FROM experiment').fetchone()
            rds_uuid, rds_url, rds_name, = self.session.execute(
                f'SELECT uuid, url, name FROM dataset where id_ = "{rds}"').fetchone()
            container.raw_dataset = DatasetInfo(rds_name, rds, rds_uuid)

            for pds_uuid, pds_url, pds_name in self.session.execute(
                    'SELECT uuid, url, name FROM dataset where type_ = "processed"').fetchall():
                container.processed_datasets.append(DatasetInfo(pds_name, (pds_url, pds_uuid), pds_uuid))
            container.keys = [key[0] for key in
                              self.session.execute('SELECT DISTINCT key FROM json_each(key_val), data').fetchall()]
            return container
        raise DataServiceError('Cannot find the experiment metadata from the given URI')

    def update_experiment(self, experiment):
        """Write an experiment to the database

        Parameters
        ----------
        experiment: Experiment
            Container of the experiment metadata

        """
        record = {
            'id_': experiment.id_,
            'uuid': experiment.uuid,
            'name': experiment.name,
            'author': experiment.author,
            'date': datetime.now() if experiment.date == 'now' else datetime.fromisoformat(experiment.date),
            'raw_dataset': experiment.raw_dataset.id_,
        }
        self.repository.save_object('Experiment', record)

    def import_data(self, experiment, data_path, name, author, format_, date='now', key_value_pairs=dict):
        """import one data to the experiment

        The data is imported to the raw dataset

        Parameters
        ----------
        experiment: Experiment
            Container of the experiment metadata
        data_path: str
            Path of the accessible data on your local computer
        name: str
            Name of the data
        author: str
            Person who created the data
        format_: str
            Format of the data (ex: tif)
        date: str
            Date when the data where created
        key_value_pairs: dict
            Dictionary {key:value, key:value} to annotate files

        Returns
        -------
        class RawData containing the metadata

        """
        self.import_glob(experiment, data_path, name, author, format_, date)

        raw_dataset_uri = os.path.abspath(experiment.raw_dataset.url[0])
        data_dir_path = os.path.join(os.path.dirname(raw_dataset_uri), experiment.raw_dataset.name)

        # create the new data uri
        data_base_name = os.path.basename(data_path)
        filtered_name = data_base_name.replace(' ', '')
        filtered_name, _ = os.path.splitext(filtered_name)

        # create the container
        metadata = RawData()
        metadata.uuid = generate_uuid()
        metadata.md_uri = (raw_dataset_uri, metadata.uuid)
        metadata.name = name
        metadata.author = author
        metadata.format = format_
        metadata.date = date
        metadata.source_dir = os.path.dirname(data_path)
        metadata.key_value_pairs = key_value_pairs
        metadata.dataset = experiment.raw_dataset.uuid

        # import data
        # print('import data with format:', metadata.format)
        # print('IMAGE ZARR ?')

        if metadata.format == 'bioformat':
            self._import_file_bioformat(raw_dataset_uri, data_path, data_dir_path, metadata.name,
                                        metadata.author, metadata.date)
        elif metadata.format == 'imagezarr':
            destination_path = os.path.join(data_dir_path, filtered_name + '.zarr')
            raw_dataset_container = self.get_dataset(raw_dataset_uri)
            raw_dataset_container.uris.append(Container(md_uri=metadata.md_uri, uuid=metadata.uuid))
            metadata.uri = destination_path
            self.update_dataset(raw_dataset_container)
            self.update_raw_data(metadata)

            self._import_file_zarr(data_path, destination_path)
            print("IMAGE ZARR !")
        else:
            format_service = formatsServices.get(metadata.format)
            files_to_copy = format_service.files(data_path)
            for file_ in files_to_copy:
                origin_base_name = os.path.basename(file_)
                destination_path = os.path.join(data_dir_path, origin_base_name)
                copyfile(file_, destination_path)
            metadata.uri = os.path.join(data_dir_path, data_base_name)  # URI is main file
            self.update_raw_data(metadata)

        # add key-value pairs to experiment
        for key in key_value_pairs:
            experiment.set_key(key)
        self.update_experiment(experiment)

        return metadata

    def _import_file_bioformat(self, raw_dataset_uri, file_path, destination_dir, data_name, author, date):
        fiji_exe = ConfigAccess.instance().get('fiji')
        cmd = f'{fiji_exe} --headless -macro bioimageit_convert.ijm "file,{file_path},' \
              f'{destination_dir},{data_name},{author},{date}"'
        print("import bioformat cmd:", cmd)
        if platform.system() == 'Windows':
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, shell=True, executable='/bin/bash', check=True)

            # add data to raw dataset
        self._add_to_raw_dataset_bioformat(destination_dir, raw_dataset_uri)

    def _import_file_zarr(self, file_path, destination_dir):
        conda_dir = ConfigAccess.instance().get('runner')['conda_dir']

        if platform.system() == 'Windows':
            condaexe = os.path.join(conda_dir, 'condabin', 'conda.bat')
            args_str = '"' + condaexe + '"' + ' activate ' + "bioimageit" + ' &&'
            cmd = f"{args_str} bioformats2raw {file_path} {destination_dir}"
            subprocess.run(cmd, check=True)
            print("import zarr image cmd:", cmd)

        else:
            condaexe = os.path.join(conda_dir, 'etc', 'profile.d', 'conda.sh')
            args_str = '"' + condaexe + '"' + ' activate ' + "bioimageit" + ' &&'
            cmd = f"{args_str} bioformats2raw {file_path} {destination_dir}"
            subprocess.run(cmd, shell=True, executable='/bin/bash', check=True)
            print("import zarr image cmd:", cmd)

    def _import_dir_bioformat(self, raw_dataset_uri, dir_uri, filter_, author, format_, date, directory_tag_key):
        fiji_exe = ConfigAccess.instance().get('fiji')

        raw_dataset_uri_ = os.path.abspath(raw_dataset_uri)
        data_dir_path = os.path.dirname(raw_dataset_uri_)

        cmd = f'{fiji_exe} --headless -macro bioimageit_convert.ijm "folder,{dir_uri},' \
              f'{data_dir_path},false,{filter_},{author},{date},{directory_tag_key}"'
        print("import bioformat cmd:", cmd)
        if platform.system() == 'Windows':
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, shell=True, executable='/bin/bash', check=True)
        self._add_to_raw_dataset_bioformat(data_dir_path, raw_dataset_uri_)

    def _add_to_raw_dataset_bioformat(self, data_dir_path, raw_dataset_uri):
        # add the data to the dataset
        tmp_file = os.path.join(data_dir_path, 'tmp.txt')
        if os.path.isfile(tmp_file):
            raw_dataset = self.get_dataset(raw_dataset_uri)
            with open(tmp_file) as file:
                lines = file.readlines()
                for line in lines:
                    line = line.rstrip('\n')

                    data_md_uri = os.path.join(data_dir_path, line)

                    # add a uuid to the data file
                    data = self.get_raw_data(data_md_uri)
                    data.uuid = generate_uuid()
                    self.update_raw_data(data)

                    data_uri = Container(
                        data_md_uri,
                        generate_uuid())

                    raw_dataset.uris.append(data_uri)
                self.update_dataset(raw_dataset)
            os.remove(tmp_file)

    def import_glob(self, experiment, files_glob, author='', format_='imagetiff', date='now', observers=None):
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
        :return: a DataFrame with the imported data
        :rtype: pandas DataFrame
        """
        files = glob.glob(files_glob)
        n = len(files)
        # Note that the code line below used to be
        # data_dir_path = os.path.join(os.path.dirname(experiment.raw_dataset.url[0]), experiment.raw_dataset.name)
        # because the url was actually the path to the database file in experiment directory. It is now the experiment
        # directory only. Maybe try and find another and better way
        experiment_path = os.path.join(ConfigAccess.instance().config['workspace'], experiment.name)
        data_dir_path = os.path.join(experiment_path, experiment.raw_dataset.name)
        date = datetime.now() if date == 'now' else datetime.fromisoformat(date),
        author = ConfigAccess.instance().config['user'] if author == '' else author
        df = pandas.DataFrame(columns=['uuid', 'name', 'dataset', 'author', 'date', 'url', 'format',
                                       'source_dir', 'meta_data', 'key_val'])
        df['author'] = pandas.Series(data=[author] * n)
        df['dataset'] = pandas.Series(data=[experiment.raw_dataset.uuid] * n)
        df['date'] = pandas.Series(data=[date[0]] * n)
        df['format'] = pandas.Series(data=[format_] * n)
        df['meta_data'] = pandas.Series(data=['{}'] * n)
        df['key_val'] = pandas.Series(data=['{}'] * n)

        for i, file in enumerate(files):
            df.loc[i, 'uuid'] = generate_uuid()
            df.loc[i, 'name'] = os.path.basename(file)
            df.loc[i, 'url'] = os.path.join(data_dir_path, df.loc[i, 'name'])
            df.loc[i, 'source_dir'] = os.path.dirname(file)

        con = self.session.connection()
        try:
            df.to_sql('data', con, if_exists='append', index=False)
            if platform.system() == 'Windows':
                os.popen(f'copy {files_glob} {data_dir_path}')
            else:
                os.popen(f'cp {files_glob} {data_dir_path}')
        except RuntimeError as error:
            raise DataServiceError('Could not import data') from error
        return df

    def import_dir(self, experiment, dir_uri, filter_, author, format_, date,
                   directory_tag_key='', observers=None):
        """Import data from a directory to the experiment

        This method import with or without copy data contained
        in a local folder into an experiment. Imported data are
        considered as RawData for the experiment

        Parameters
        ----------
        experiment: Experiment
            Container of the experiment metadata
        dir_uri: str
            URI of the directory containing the data to be imported
        filter_: str
            Regular expression to filter which files in the folder
            to import
        author: str
            Name of the person who created the data
        format_: str
            Format of the image (ex: tif)
        date: str
            Date when the data where created
        directory_tag_key
            If the string directory_tag_key is not empty, a new tag key entry with the
            key={directory_tag_key} and the value={the directory name}.
        observers: list
            List of observers to notify the progress

        """
        files = os.listdir(dir_uri)
        count = 0
        key_value_pairs = {}
        if directory_tag_key != '':
            # key_value_pairs[directory_tag_key] = os.path.dirname(dir_uri)
            key_value_pairs[directory_tag_key] = dir_uri

        if format_ == 'bioformat':
            self._import_dir_bioformat(experiment.raw_dataset.md_uri, dir_uri, filter_,
                                       author, format_, date, directory_tag_key)
        else:
            for file in files:
                count += 1
                r1 = re.compile(filter_)
                if r1.search(file):
                    if observers is not None:
                        for obs in observers:
                            obs.notify_progress(int(100 * count / len(files)), file)
                    self.import_data(experiment, os.path.join(dir_uri, file), file, author,
                                     format_, date, key_value_pairs)

    def get_raw_data(self, md_uri):
        return self._get_data(md_uri, RawData)

    def _get_data(self, md_uri, cls=RawData):
        """Read data from the database

        Parameters
        ----------
        md_uri: str
            URI if the data
        Returns
        -------
        RawData object containing the data metadata

        """
        try:
            container = cls()
            container.md_uri = md_uri
            statement = sqlalchemy.text("""
                SELECT 
                    data.uuid,
                    data.name, 
                    data.dataset,
                    data.format,
                    data.author,
                    data.date,
                    data.url,
                    data.source_dir,
                    data.meta_data,
                    data.key_val 
                FROM data
                WHERE data.uuid = :uuid
                """)
            container.uuid, container.name, container.dataset, \
            container.format, container.author, \
            container.date, container.uri, \
            container.source_dir, \
            metadata, key_value_pairs = self.session.connection().execute(statement, uuid=md_uri[1]).fetchone()
            # metadata
            container.metadata = json.loads(metadata)
            container.key_value_pairs = json.loads(key_value_pairs)

            return container
        except RuntimeError as error:
            raise DataServiceError(f'Error getting data from database: {md_uri[1]}') from error

    def update_raw_data(self, raw_data):
        """Read a raw data from the database

        Parameters
        ----------
        raw_data: RawData
            Container with the raw data metadata

        """
        raw_data.type = METADATA_TYPE_RAW
        try:
            statement = sqlalchemy.text("""
            INSERT OR IGNORE INTO data 
            VALUES (:uuid, :name, :dataset, :author, :date, :uri, :format_, :source_dir, :metadata, :key_val)
            """)
            self.session.connection().execute(statement,
                                              uuid=raw_data.uuid,
                                              name=raw_data.name,
                                              dataset=raw_data.dataset,
                                              author=raw_data.author,
                                              date=raw_data.date,
                                              uri=raw_data.uri,
                                              format_=raw_data.format,
                                              source_dir=raw_data.source_dir,
                                              metadata=str(raw_data.metadata),
                                              key_val=json.dumps(raw_data.key_value_pairs))

            statement = sqlalchemy.text(
                """
UPDATE data
SET (uuid, name, dataset, author, date, url, format, source_dir, meta_data, key_val) =
(:uuid, :name, :dataset, :author, :date, :uri, :format_, :source_dir, :metadata, :key_val)
WHERE uuid = :uuid
            """)
            self.session.connection().execute(statement,
                                              uuid=raw_data.uuid,
                                              name=raw_data.name,
                                              dataset=raw_data.dataset,
                                              author=raw_data.author,
                                              date=raw_data.date,
                                              uri=raw_data.uri,
                                              format_=raw_data.format,
                                              source_dir=raw_data.source_dir,
                                              metadata=str(raw_data.metadata),
                                              key_val=json.dumps(raw_data.key_value_pairs))
        except RuntimeError as error:
            raise DataServiceError('Could not update raw data') from error

    def get_processed_data(self, md_uri):
        """Read a processed data from the database

        Parameters
        ----------
        md_uri: str
            URI if the processed data

        Returns
        -------
        ProcessedData object containing the raw data metadata

        """
        container = self._get_data(md_uri, ProcessedData)

        if container is not None:
            # origin run
            statement = sqlalchemy.text("""
                SELECT dataset.run FROM data, dataset
                WHERE data.uuid = :uuid AND data.dataset = dataset.uuid
                """)
            run_uuid = self.session.connection().execute(statement, uuid=md_uri[1]).fetchone()[0]
            container.run = run_uuid
            # container.run = self.get_run((md_uri[0], run_uuid))

            return container
        # raise DataServiceError(f'Metadata file format not supported {md_uri}')
        return None

    def update_processed_data(self, processed_data):
        """Read a processed data from the database

        Parameters
        ----------
        processed_data: ProcessedData
            Container with the processed data metadata

        """
        try:
            statement = sqlalchemy.text("""
                        INSERT OR IGNORE INTO data 
                        VALUES (:uuid, :name, :dataset, :author, :date, :uri, :format_, :source_dir, :metadata, :key_val)
                        """)
            self.session.execute(self.session.connection().execute(statement,
                                                                   uuid=processed_data.uuid,
                                                                   name=processed_data.name,
                                                                   dataset=processed_data.dataset,
                                                                   author=processed_data.author,
                                                                   date=processed_data.date,
                                                                   uri=processed_data.uri,
                                                                   format_=processed_data.format,
                                                                   source_dir=processed_data.source_dir,
                                                                   metadata=str(processed_data.metadata),
                                                                   key_val=json.dumps(processed_data.key_value_pairs))
                                 )
            statement = sqlalchemy.text(
                """
UPDATE data
SET (uuid, name, dataset, author, date, url, format, source_dir, meta_data, key_val) =
(:uuid, :name, :dataset, :author, :date, :uri, :format_, :source_dir, :metadata, :key_val)
WHERE uuid = :uuid
            """)
            self.session.connection().execute(statement,
                                              uuid=processed_data.uuid,
                                              name=processed_data.name,
                                              dataset=processed_data.dataset,
                                              author=processed_data.author,
                                              date=processed_data.date,
                                              uri=processed_data.uri,
                                              format_=processed_data.format,
                                              source_dir=processed_data.source_dir,
                                              metadata=str(processed_data.metadata),
                                              key_val=json.dumps(processed_data.key_value_pairs))
        except RuntimeError as error:
            raise DataServiceError('Could not update processed data') from error

        dataset = self.get_dataset((processed_data.md_uri[0], processed_data.dataset))
        dataset.type = METADATA_TYPE_PROCESSED
        dataset.run = processed_data.run.uuid

        self.update_dataset(dataset)

    def get_dataset(self, md_uri):
        """Read a dataset from the database using it URI

        Parameters
        ----------
        md_uri: str
            URI if the dataset

        Returns
        -------
        Dataset object containing the dataset metadata

        """
        record = self.repository.get_record('Dataset', md_uri)
        container = Dataset()
        container.id_ = record['id_']
        container.uuid = record['uuid']
        container.name = record['name']
        container.url = os.path.join(record['url'], record['name'])
        container.type = record['type_']
        container.run = record['run']
        container.md_uri = container.url

        for id_, uuid in self.session.execute(f'SELECT id_, uuid FROM data where dataset = "{container.uuid}"'):
            container.uris.append(Container(id_, uuid))

        return container


    def update_dataset(self, dataset):
        """Read a processed data from the database

        Parameters
        ----------
        dataset: Dataset
            Container with the dataset metadata

        """
        record = {
            'id_': dataset.id_,
            'uuid': dataset.uuid,
            'name': dataset.name,
            'url': dataset.url,
            'type_': dataset.type,
            'run': dataset.run,
        }
        return self.repository.save_object('Dataset', record)

    def create_dataset(self, experiment, dataset_name):
        """Create a processed dataset in an experiment

        Parameters
        ----------
        experiment: Experiment
            Object containing the experiment metadata
        dataset_name: str
            Name of the dataset

        Returns
        -------
        Dataset object containing the new dataset metadata

        """
        # create the dataset metadata
        experiment_dir = os.path.dirname(experiment.md_uri)
        dataset_dir = os.path.join(experiment_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            os.mkdir(dataset_dir)
        container = Dataset()
        container.uuid = generate_uuid()
        container.md_uri = (experiment.md_uri, container.uuid)
        container.name = dataset_name
        container.type = METADATA_TYPE_PROCESSED
        container.run = ''

        self.update_dataset(container)

        # add the dataset to the experiment
        experiment.processed_datasets.append(
            DatasetInfo(dataset_name, container.md_uri, container.uuid)
        )
        self.update_experiment(experiment)

        return container

    def create_run(self, dataset, run_info):
        """Create a new run metadata

        Parameters
        ----------
        dataset: Dataset
            Object of the dataset metadata
        run_info: Run
            Object containing the metadata of the run. md_uri is ignored and
            created automatically by this method

        Returns
        -------
        Run object with the metadata and the new created md_uri

        """
        # create run URI
        # dataset_md_uri = os.path.abspath(dataset.md_uri)
        # dataset_dir = SQLiteMetadataService.md_file_path(dataset_md_uri)
        # run_md_file_name = "run.md.json"
        # run_id_count = 0
        # while os.path.isfile(os.path.join(dataset_dir, run_md_file_name)):
        #     run_id_count += 1
        #     run_md_file_name = "run_" + str(run_id_count) + ".md.json"
        # run_uri = os.path.join(dataset_dir, run_md_file_name)

        # write run
        run_info.processed_dataset = dataset
        run_info.uuid = generate_uuid()
        run_info.md_uri = (dataset.md_uri[0], run_info.uuid)
        run_info.uri = os.path.join(os.path.dirname(dataset.md_uri[0]), dataset.name)
        self._write_run(run_info)
        return run_info

    def get_dataset_runs(self, dataset):
        """Read the run metadata from a dataset

        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        List of Runs

        """
        run_uri = (dataset.md_uri[0], dataset.run)
        return [self.get_run(run_uri)]

    def get_run(self, md_uri):
        """Read a run metadata from the data base

        Parameters
        ----------
        md_uri
            URI of the run entry in the database

        Returns
        -------
        Run: object containing the run metadata

        """
        try:
            container = Run()

            container.uuid, container.process_name, process_uri, inputs, parameters = self.session.execute(
                f'SELECT * FROM run where uuid = "{md_uri[1]}"').fetchone()
            container.md_uri = md_uri
            container.process_uri = SQLiteMetadataService.normalize_path_sep(process_uri)
            container.inputs = json.loads(inputs)
            container.parameters = json.loads(parameters)

            ds_uuid = self.session.execute(f'SELECT uuid FROM dataset where run = "{container.uuid}"').fetchone()[0]
            container.processed_dataset = self.get_dataset((md_uri[0], ds_uuid))
            container.uri = os.path.join(os.path.dirname(md_uri[0]), container.processed_dataset.name)
            # container.processed_dataset = Container(
            #     SQLiteMetadataService.absolute_path(
            #         SQLiteMetadataService.normalize_path_sep(
            #             metadata['processed_dataset']['url']),
            #         md_uri),
            #     metadata['processed_dataset']['uuid']
            # )
            return container
        except RuntimeError as error:
            raise DataServiceError('Run not found') from error

    def _write_run(self, run):
        """Write a run metadata to the data base

        Parameters
        ----------
        run
            Object containing the run metadata

        """
        try:
            run_process_url = SQLiteMetadataService.to_unix_path(run.process_uri)
            run_inputs = [{
                'name': input_.name,
                'dataset': input_.dataset,
                'query': input_.query,
                'origin_output_name': input_.origin_output_name,
            } for input_ in run.inputs]

            run_parameters = [{'name': parameter.name, 'value': parameter.value} for parameter in run.parameters]

            statement = sqlalchemy.text(
                'INSERT OR IGNORE INTO run VALUES (:uuid, :process_name, :process_url, :inputs, :parameters)'
            )
            self.session.connection().execute(statement,
                                              uuid=run.uuid,
                                              process_name=run.process_name,
                                              process_url=run_process_url,
                                              inputs=json.dumps(run_inputs),
                                              parameters=json.dumps(run_parameters))

            statement = sqlalchemy.text(
                """
UPDATE run
SET (uuid, process_name, process_url, inputs, parameters) = (:uuid, :process_name, :process_url, :inputs, :parameters)
WHERE uuid = :uuid
                """
            )
            self.session.connection().execute(statement,
                                              uuid=run.uuid,
                                              process_name=run.process_name,
                                              process_url=run_process_url,
                                              inputs=json.dumps(run_inputs),
                                              parameters=json.dumps(run_parameters))
        except RuntimeError as error:
            raise DataServiceError('Could not write run metadata') from error

        # metadata['processed_dataset'] = {"uuid": run.processed_dataset.uuid,
        #                                  "url": dataset_rel_url}
        # metadata['inputs'] = []
        # for input_ in run.inputs:
        #     metadata['inputs'].append(
        #         {
        #             'name': input_.name,
        #             'dataset': input_.dataset,
        #             'query': input_.query,
        #             'origin_output_name': input_.origin_output_name,
        #         }
        #     )
        # metadata['parameters'] = []
        # for parameter in run.parameters:
        #     metadata['parameters'].append(
        #         {'name': parameter.name, 'value': parameter.value}
        #     )

    def get_data_uri(self, data_container):
        return data_container.uri.replace('\\', '\\\\')

    def create_data_uri(self, dataset, run, processed_data):
        """Create the URI of the new data

        Parameters
        ----------
        dataset: Dataset
            Object of the dataset metadata
        run: Run
            Metadata of the run
        processed_data: ProcessedData
            Object containing the new processed data. md_uri is ignored and
            created automatically by this method

        Returns
        -------
        ProcessedData object with the metadata and the new created md_uri

        """
        dataset_dir = os.path.dirname(os.path.abspath(dataset.md_uri[0]))

        extension = FormatsAccess.instance().get(processed_data.format).extension
        processed_data.uri = os.path.join(dataset_dir,
                                          dataset.name, f"{processed_data.name}.{extension}").replace('\\', '\\\\')
        return processed_data

    def create_data(self, dataset, run, processed_data):
        """Create a new processed data for a given dataset

        Parameters
        ----------
        dataset: Dataset
            Object of the dataset metadata
        run: Run
            Metadata of the run
        processed_data: ProcessedData
            Object containing the new processed data. md_uri is ignored and
            created automatically by this method

        Returns
        -------
        ProcessedData object with the metadata and the new created md_uri

        """
        dataset_dir = os.path.dirname(dataset.md_uri[0])

        # create the data metadata
        processed_data.uuid = generate_uuid()
        processed_data.md_uri = (dataset.md_uri[0], processed_data.uuid)
        extension = FormatsAccess.instance().get(processed_data.format).extension
        processed_data.uri = os.path.join(dataset_dir, dataset.name, f"{processed_data.name}.{extension}")

        processed_data.run = run
        processed_data.dataset = dataset.uuid

        processed_data.source_dir = None
        processed_data.metadata = {}
        processed_data.key_value_pairs = {}

        self.update_processed_data(processed_data)
        #
        # # add the data to the dataset
        # dataset.uris.append(Container(data_md_file, processed_data.uuid))
        # self.update_dataset(dataset)

        return processed_data

    def download_data(self, md_uri, destination_file_uri):
        if destination_file_uri == '':
            raw_data = self.get_raw_data(md_uri)
            return raw_data.uri
        return destination_file_uri

    def view_data(self, md_uri):
        raw_data = self.get_raw_data(md_uri)
        dir(raw_data)
        if raw_data.format == 'imagetiff':
            return imread(raw_data.uri)
        if raw_data.format == 'imagezarr':
            return zarr.open(os.path.join(raw_data.uri, "0", "0"), mode='r')
        return None
