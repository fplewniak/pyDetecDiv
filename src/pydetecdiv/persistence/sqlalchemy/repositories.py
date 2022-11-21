#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Concrete Repositories using a SQL database with the sqlalchemy toolkit
"""
import json
import os
import re

import pandas
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import Delete
from pandas import DataFrame
from bioimageit_core.plugins.data_factory import metadataServices
from pydetecdiv.persistence.repository import ShallowDb
from pydetecdiv.persistence.sqlalchemy.orm.main import mapper_registry
from pydetecdiv.persistence.sqlalchemy.orm.dao import dso_dao_mapping as dao
from pydetecdiv.persistence.sqlalchemy.orm.associations import Linker
from pydetecdiv.persistence.bioimageit.request import Request
from pydetecdiv.persistence.bioimageit.plugins.data_sqlite import SQLiteMetadataServiceBuilder
from pydetecdiv.settings import get_config_value


class ShallowSQLite3(ShallowDb):
    """
    A concrete shallow SQLite3 persistence inheriting ShallowDb and implementing SQLite3-specific engine
    """

    def __init__(self, dbname=None):
        self.name = dbname
        self.engine = sqlalchemy.create_engine(f'sqlite+pysqlite:///{self.name}')
        self.session_maker = sessionmaker(self.engine, future=True)
        self.session_ = None

        self.bioiit_req = Request(get_config_value('bioimageit', 'config_file'), debug=False)
        metadataServices.register_builder('SQLITE', SQLiteMetadataServiceBuilder())
        self.bioiit_req.connect()
        self.bioiit_req.data_service.connect_to_session(self.session)
        self.bioiit_exp = None

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
            self.bioiit_exp = self.bioiit_req.create_experiment(exp_name)
        else:
            self.bioiit_exp = self.bioiit_req.get_experiment(self.name)

    def close(self):
        """
        Close the current connexion
        """
        self.engine.dispose()

    def import_images(self, source_path):
        """
        Import images from a source path. All files corresponding to the path will be imported.
        :param source_path: the source path (glob pattern)
        :type source_path: str
        """
        self.bioiit_req.import_glob(self.bioiit_exp, source_path)

    def determine_fov(self, source, regex):
        return self.determine_links_using_regex('data', source, ('FOV',), regex)

    def determine_links_using_regex(self, dataset_name, source, keys_, regex):
        dataset = self.bioiit_req.get_dataset(self.bioiit_exp, dataset_name)
        df = self.bioiit_req.data_service.determine_links_using_regex(dataset, source, keys_, regex)
        return df

    def annotate_raw_data(self, source, keys_, regex):
        """
        Returns a DataFrame containing all the metadata associated to raw data, including annotations created using a
        regular expression applied to a field or a combination thereof.
        :param source: the database field or combination of fields to apply the regular expression to
        :type source: str or callable returning a str
        :param keys_: the list of classes created objects belong to
        :type keys_: tuple of str
        :param regex: regular expression defining the DSOs' names
        :type regex: regular expression str
        :return: a table of all the metadata associated to raw data
        :rtype: pandas DataFrame
        """
        dataset = self.bioiit_req.get_dataset(self.bioiit_exp, 'data')
        df = self.bioiit_req.data_service.create_annotations_using_regex(self.bioiit_exp, dataset, source, keys_, regex)
        return pandas.DataFrame([json.loads(k) for k in df.key_val]).join(df.drop(labels='key_val', axis=1))

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

    def get_record(self, class_name, id_=None):
        """
        A method returning an object record of a given class from its id
        :param class_name: the class name of object to get the record of
        :type class_name: str
        :param id_: the id of the requested object
        :type id_: int
        :return: the object record
        :rtype: dict (record)
        """
        return self.session.get(dao[class_name], id_).record

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
        linked_records = []
        if cls_name == 'ImageData':
            if parent_cls_name in ['Image']:
                linked_records = [self.get_record(cls_name, self.get_record(parent_cls_name, parent_id)['image_data'])]
            if parent_cls_name in ['FOV', 'ROI']:
                linked_records = dao[parent_cls_name](self.session).image_data(parent_id)
        if cls_name == 'FOV':
            if parent_cls_name in ['ROI']:
                linked_records = [self.get_record(cls_name, self.get_record(parent_cls_name, parent_id)['fov'])]
            if parent_cls_name in ['Image', 'ImageData', ]:
                linked_records = dao[parent_cls_name](self.session).fov(parent_id)
            if parent_cls_name in ['data', ]:
                linked_records = dao[parent_cls_name](self.session).fov_list(parent_id)
        if cls_name == 'ROI':
            if parent_cls_name in ['ImageData', ]:
                linked_records = [self.get_record(cls_name, self.get_record(parent_cls_name, parent_id)['roi'])]
            if parent_cls_name in ['FOV', ]:
                linked_records = dao[parent_cls_name](self.session).roi_list(parent_id)
            if parent_cls_name in ['Image', ]:
                linked_records = dao[parent_cls_name](self.session).roi(parent_id)
        if cls_name == 'Image':
            if parent_cls_name in ['ImageData', 'FOV', 'ROI']:
                linked_records = dao[parent_cls_name](self.session).image_list(parent_id)
        if cls_name == 'Data':
            if parent_cls_name in ['FOV']:
                linked_records = dao[parent_cls_name](self.session).data(parent_id)

        return linked_records

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
        obj1 = self.session.get(dao[class1_name], id_1)
        obj2 = self.session.get(dao[class2_name], id_2)
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
