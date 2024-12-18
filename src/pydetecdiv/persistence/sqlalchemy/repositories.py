#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Concrete Repositories using a SQL database with the sqlalchemy toolkit
"""
import json
import os
import re
import sqlite3
import subprocess
from datetime import datetime
from PIL import Image

import pandas
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import Delete
from pandas import DataFrame
from pydetecdiv.persistence.repository import ShallowDb
from pydetecdiv.persistence.sqlalchemy.orm.main import mapper_registry
from pydetecdiv.persistence.sqlalchemy.orm.dao import dso_dao_mapping as dao
from pydetecdiv.persistence.sqlalchemy.orm.associations import Linker
from pydetecdiv.settings import get_config_value
from pydetecdiv import generate_uuid, copy_files
from pydetecdiv.exceptions import OpenProjectError, ImportImagesError


class ShallowSQLite3(ShallowDb):
    """
    A concrete shallow SQLite3 persistence inheriting ShallowDb and implementing SQLite3-specific engine
    """

    def __init__(self, dbname=None):
        self.name = dbname
        try:
            self.engine = sqlalchemy.create_engine(f'sqlite+pysqlite:///{self.name}?check_same_thread=True')
        except sqlite3.OperationalError as e:
            raise OpenProjectError(f'Could not open project\n{e.message}') from e
        self.session_maker = sessionmaker(self.engine, future=True)
        self.session_ = None
        self.create()

    @property
    def session(self):
        """
        Property returning the sqlalchemy Session object attached to the repository

        :return:
        """
        if self.session_ is None:
            self.session_ = self.session_maker()
        return self.session_

    def commit(self):
        """
        Commit the current transaction
        """
        if self.session_ is not None:
            self.session_.commit()

    def rollback(self):
        """
        Rollback the current transaction
        """
        if self.session_ is not None:
            self.session_.rollback()

    def executescript(self, script):
        """
        Reads a string containing several SQL statements in a free format

        :param script: the string representing the SQL script to be executed
        :type script: str
        """
        try:
            statements = re.split(r';\s*$', script, flags=re.MULTILINE)
            for statement in statements:
                if statement:
                    self.session.execute(sqlalchemy.text(statement))
        except sqlalchemy.exc.OperationalError as exc:
            print(exc)

    def create(self):
        """
        Gets SqlAlchemy classes defining the project database schema and creates the database if it does not exist.
        """
        exp_name = str(os.path.splitext(os.path.basename(self.name))[0])
        if not sqlalchemy.inspect(self.engine).get_table_names():
            mapper_registry.metadata.create_all(self.engine)
            experiment_path = os.path.join(get_config_value('project', 'workspace'), exp_name)
            os.mkdir(experiment_path)
            dataset_record = {'id_': None,
                              'uuid': generate_uuid(),
                              'name': 'data',
                              'url': experiment_path,
                              'type_': 'raw',
                              'run': None,
                              'pattern': None,
                              }
            self.save_object('Dataset', dataset_record)
            os.mkdir(os.path.join(experiment_path, dataset_record['name']))
            experiment_record = {'id_': None,
                                 'uuid': generate_uuid(),
                                 'name': exp_name,
                                 'author': get_config_value('project', 'user'),
                                 'date': datetime.now(),
                                 'raw_dataset': self.get_record_by_name('Dataset', dataset_record['name'])['id_'],
                                 }
            self.save_object('Experiment', experiment_record)
            self.commit()

    def close(self):
        """
        Close the current connexion
        """
        self.engine.dispose()

    def import_images(self, image_files, data_dir_path, destination, author='', date='now', in_place=False,
                      img_format='imagetiff') -> subprocess.Popen:
        """
        Import images specified in a list of files into a destination

        :param image_files: list of image files to import
        :type image_files: list of str
        :param data_dir_path: path for the current project raw data directory
        :type data_dir_path: path or str
        :param destination: destination directory to import files into
        :type destination: str
        :param author: the user importing the data
        :type author: str
        :param date: the date of import
        :type date: str
        :param in_place: boolean indicating whether image files should be copied (False) or kept in place (True)
        :type in_place: bool
        :param img_format: the file format
        :type img_format: str
        :return: the list of imported files. This list can be used to roll the copy back if needed
        :rtype: list of str
        """
        # urls = []
        if destination:
            data_dir_path = os.path.join(data_dir_path, destination)
        try:
            process = copy_files(image_files, data_dir_path) if not in_place else None
            for image_file in image_files:
                record = {
                    'id_': None,
                    'uuid': generate_uuid(),
                    'name': os.path.basename(image_file),
                    'dataset': self.get_record_by_name('Dataset', 'data')['id_'],
                    'author': get_config_value('project', 'user') if author == '' else author,
                    'date': datetime.now() if date == 'now' else datetime.fromisoformat(date),
                    'url': image_file if in_place else os.path.join(destination, os.path.basename(image_file)),
                    'format': img_format,
                    'source_dir': os.path.dirname(image_file),
                    'meta_data': '{}',
                    'key_val': '{}',
                }
                with Image.open(record['url']) as img:
                    record['xdim'], record['ydim'] = img.size
                self.save_object('Data', record)
                # urls.append(record['url'])
        except:
            raise ImportImagesError('Could not import images')
        # return urls, process
        return process

    def annotate_data(self, dataset, source, keys_, regex):
        """
        Method to annotate data files in a dataset according to a regular expression applied to a source. The resulting
        key-value pairs are placed in a key_val column.

        :param dataset: the dataset whose data should be annotated
        :type dataset: Dataset object
        :param source: the database field or combination of fields to apply the regular expression to
        :type source: str or callable returning a str
        :param keys_: the list of classes created objects belong to
        :type keys_: tuple of str
        :param regex: regular expression defining the annotations
        :type regex: regular expression str
        :return: list of annotated Data records in a dataframe
        :rtype: pandas.DataFrame
        """
        data_list = self.session.query(dao['Dataset']).filter(dao['Dataset'].id_ == dataset.id_).first().data_list_

        pattern = re.compile(regex)

        call_back = source if callable(source) else lambda x: x.record[source]

        df = pandas.DataFrame([d.record for d in data_list])
        for i, data in enumerate(data_list):
            m = re.search(pattern, call_back(data))
            if m:
                # key_val = json.loads(data.key_val)
                # key_val.update(dict(zip(keys_, [m.group(k) for k in keys_])))
                # data.key_val = json.dumps(key_val)
                for k in keys_:
                    df.loc[i, k] = m.group(k)
        return df

    def save_object(self, class_name, record):
        """
        Save the object represented by the record

        :param class_name: the class name of the object to save into SQL database
        :type class_name: str
        :param record: the record representing the object
        :type record: dict
        :return: the id of the created or updated object
        :rtype: int
        """
        if record['id_'] is None:
            return dao[class_name](self.session).insert(record)
        return dao[class_name](self.session).update(record)

    def delete_object(self, class_name, id_):
        """
        Delete an object of class name = class_name with id = id_

        :param class_name: the class name of the object to delete
        :type class_name: str
        :param id_: the id of the object to delete
        :type id_: int
        """
        self.session.execute(Delete(dao[class_name], whereclause=dao[class_name].id_ == id_))
        # self.session.commit()

    def _get_records(self, class_name=None, query=None):
        """
        A private method returning the list of all object records of a given class specified by its name and verifying a
        query built from a list of where clauses

        :param class_name: the name of the class of objects to get records of
        :param query: a list of sqlalchemy where clauses defining the selection SQL query
        :type class_name: str
        :type query: list of where clauses
        :return: a list of records
        :rtype: list of dictionaries (records)
        """
        dao_list = self.session.query(dao[class_name])
        if query is not None:
            for q in query:
                dao_list = dao_list.where(q)

        return [obj.record for obj in dao_list]

    def count_records(self, class_name):
        """
        Get the number of objects of a given class in the current project

        :param class_name: the class name of the objects whose count will be returned
        :type class_name: str
        :return: the number of objects
        :rtype: int
        """
        return self.session.query(dao[class_name]).count()

    def get_dataframe(self, class_name, id_list=None):
        """
        Get a DataFrame containing the list of all domain objects of a given class in the current project

        :param class_name: the class name of the objects whose list will be returned
        :type class_name: str
        :param id_list: the list of ids of objects to retrieve
        :type id_list: a list of int
        :return: a DataFrame containing the list of objects
        :rtype: DataFrame containing the records representing the requested domain-specific objects
        """
        return DataFrame(self.get_records(class_name, id_list))

    def get_record(self, class_name, id_=None, uuid=None):
        """
        A method returning an object record of a given class from its id

        :param class_name: the class name of object to get the record of
        :type class_name: str
        :param id_: the id of the requested object
        :type id_: int
        :return: the object record
        :rtype: dict (record)
        """
        if id_ is not None:
            dso = self.session.get(dao[class_name], id_)
            if dso is not None:
                return dso.record
        if uuid is not None:
            return self.session.query(dao[class_name]).filter(dao[class_name].uuid == uuid).first().record
        return None

    def get_record_by_name(self, class_name, name=None):
        """
        Return a record from its name

        :param class_name: class name of the corresponding DSO object
        :type class_name: str
        :param name: the name of the requested record
        :type name: str
        :return: the record
        :rtype: dict
        """
        obj = self.session.query(dao[class_name]).where(dao[class_name].name.in_([name])).first()
        if obj:
            return obj.record
        return None

    def get_records(self, class_name, id_list=None):
        """
        A method returning the list of all object records of a given class or select those whose id is in id_list

        :param class_name: the class name of objects to get records of
        :type class_name: str
        :param id_list: the list of ids of objects to retrieve
        :type id_list: a list of int
        :return: a list of records
        :rtype: list of dictionaries (records)
        """
        if id_list is not None:
            dao_list = self.session.query(dao[class_name]).where(dao[class_name].id_.in_(id_list))
        else:
            dao_list = self.session.query(dao[class_name])
        return [obj.record for obj in dao_list]

    def get_linked_records(self, cls_name, parent_cls_name, parent_id):
        """
        A method returning the list of records for all objects of class defined by cls_name that are linked to object
        of class parent_cls_name with id_ = parent_id

        :param cls_name: the class name of the objects to retrieve records for
        :type cls_name: str
        :param parent_cls_name: the class nae of the parent object
        :type parent_cls_name: str
        :param parent_id: the id of the parent object
        :type parent_id: int
        :return: a list of records
        :rtype: list of dict
        """
        match (cls_name, parent_cls_name):
            # case ['ImageData', 'Image']:
            #     linked_rec = [self.get_record(cls_name, self.get_record(parent_cls_name, parent_id)['image_data'])]
            # case ['ImageData', ('FOV' | 'ROI')]:
            #     linked_rec = dao[parent_cls_name](self.session).image_data(parent_id)
            case ['FOV', ('ROI' | 'ImageResource')]:
                linked_rec = [self.get_record(cls_name, self.get_record(parent_cls_name, parent_id)['fov'])]
            # case ['FOV', ('Image' | 'ImageData')]:
            #     linked_rec = dao[parent_cls_name](self.session).fov(parent_id)
            case ['FOV', 'Data']:
                linked_rec = dao[parent_cls_name](self.session).fov_list(parent_id)
            # case ['ROI', 'ImageData']:
            #     linked_rec = [self.get_record(cls_name, self.get_record(parent_cls_name, parent_id)['roi'])]
            case ['ROI', ('FOV' | 'Data')]:
                linked_rec = dao[parent_cls_name](self.session).roi_list(parent_id)
            # case ['ROI', 'Image']:
            #     linked_rec = dao[parent_cls_name](self.session).roi(parent_id)
            # case ['Image', ('ImageData' | 'FOV' | 'ROI')]:
            #     linked_rec = dao[parent_cls_name](self.session).image_list(parent_id)
            case ['Data', ('FOV' | 'ROI')]:
                linked_rec = dao[parent_cls_name](self.session).data(parent_id)
            case ['Data', ('Dataset' | 'ImageResource')]:
                linked_rec = dao[parent_cls_name](self.session).data_list(parent_id)
            case ['Dataset', 'Data']:
                linked_rec = [self.get_record(cls_name, self.get_record(parent_cls_name, parent_id)['dataset'])]
            case ['ImageResource', 'Data']:
                linked_rec = [self.get_record(cls_name, self.get_record(parent_cls_name, parent_id)['image_resource'])]
            case ['ImageResource', 'FOV']:
                linked_rec = dao[parent_cls_name](self.session).image_resources(parent_id)
            case _:
                linked_rec = []
        return linked_rec

    def _get_dao(self, class_name, id_=None):
        """
        A method returning a DAO of a given class from its id

        :param class_name: the class name of object to get the record of
        :type class_name: str
        :param id_: the id of the requested object
        :type id_: int
        :return: the DAO
        :rtype: object
        """
        obj = self.session.get(dao[class_name], id_)
        obj.session = self.session
        return obj

    def link(self, class1_name, id_1, class2_name, id_2):
        """
        Create a link between two domain-specific objects. There must be a direct link defined in Linker class,
        otherwise, the link cannot be created.

        :param class1_name: the class name of the first object to link
        :type class1_name: str
        :param id_1: the id of the first object to link
        :type id_1: int
        :param class2_name: the class name of the second object to link
        :type class2_name: str
        :param id_2: the id of the second object to link
        :type id_2: int
        """
        obj1 = self._get_dao(class1_name, id_1)
        obj2 = self._get_dao(class2_name, id_2)
        Linker.link(obj1, obj2)

    def unlink(self, class1_name, id_1, class2_name, id_2):
        """
        Remove the link between two domain-specific objects. There must be a direct link defined in Linker class,
        otherwise, the link cannot be removed.

        :param class1_name: the class name of the first object to unlink
        :type class1_name: str
        :param id_1: the id of the first object to unlink
        :type id_1: int
        :param class2_name: the class name of the second object to unlink
        :type class2_name: str
        :param id_2: the id of the second object to unlink
        :type id_2: int
        """
        dao1_class, dao2_class = dao[class1_name].__name__, dao[class2_name].__name__
        self.session.delete(self.session.get(Linker.association(dao1_class, dao2_class), (id_1, id_2)))
