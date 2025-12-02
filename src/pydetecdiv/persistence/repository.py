#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Definition of the Repository interface accessible from the Business-logic layer and that must be implemented by
concrete repositories.
"""
import abc
import subprocess
from datetime import datetime
from typing import Any

import pandas


class ShallowDb(abc.ABC):
    """
    Abstract class used as an interface to encapsulate project persistence access
    """

    @abc.abstractmethod
    def create(self) -> None:
        """
        Abstract method enforcing the implementation in all shallow persistence connectors of a create() method for
        creating the database if it does not exist
        """

    @abc.abstractmethod
    def commit(self) -> None:
        """
        Abstract method enforcing the implementation of a method to save creations and updates of objects in repository
        on disk
        """

    @abc.abstractmethod
    def rollback(self) -> None:
        """
        Abstract method enforcing the implementation of a method cancelling operations that have not been committed
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Abstract method enforcing the implementation of a close() method in all shallow persistence connectors
        """

    @abc.abstractmethod
    def import_images(self, image_files: list[str], data_dir_path: str, destination: str, author: str = '', date: datetime = 'now',
                      in_place: bool = False, img_format: str = 'imagetiff') -> subprocess.Popen:
        """
        Import images specified in a list of files into a destination

        :param image_files: list of image files to import
        :param data_dir_path: path for the current project raw data directory
        :param destination: destination directory to import files into
        :param author: the user importing the data
        :param date: the date of import
        :param in_place: boolean indicating whether image files should be copied (False) or kept in place (True)
        :param img_format: the file format
        :return: the list of imported files. This list can be used to roll the copy back if needed
        """

    @abc.abstractmethod
    def save_object(self, class_name: str, record: dict[str, Any]) -> int:
        """
        Abstract method enforcing the implementation of a save_object() method in all shallow persistence connectors

        :param class_name: the class name of the object to save
        :param record: the record representing the object
        """

    @abc.abstractmethod
    def delete_object(self, class_name: str, id_: int) -> None:
        """
        Abstract method enforcing the implementation of a delete_object() method in all shallow persistence connectors

        :param class_name: the class name of the object to delete
        :param id_: the id of the object to delete
        """

    @abc.abstractmethod
    def count_records(self, class_name: str) -> int:
        """
        Get the number of objects of a given class in the current project

        :param class_name: the class name of the objects whose count will be returned
        :return: the number of objects
        """

    @abc.abstractmethod
    def get_dataframe(self, class_name: str, id_list: list[int] = None) -> pandas.DataFrame:
        """
        Abstract method enforcing the implementation of a method returning a pandas DataFrame containing the list of
        objects of a given class in the current project whose id is in id_list

        :param class_name: the class name of the objects whose list will be returned
        :param id_list: the list of ids of objects to retrieve
        :return: a DataFrame containing the list of objects
        """

    @abc.abstractmethod
    def get_record(self, class_name: str, id_: int = None, uuid: str = None) -> dict[str, Any] | None:
        """
        Abstract method enforcing the implementation of a method returning an object record of a given
        class whose id is id_

        :param class_name: the class of the requested object
        :param id_: the id of the requested object
        :return: a record (i.e. dictionary) containing the data for the requested object
        """

    @abc.abstractmethod
    def get_records(self, class_name: str, id_list: list[int] = None) -> list[dict[str, Any]]:
        """
        Abstract method enforcing the implementation of a method returning the list of all object records of a given
        class specified by its name or a list of objects whose id is in id_list

        :param class_name: the class name of the requested objects
        :param id_list: the list of ids for objects
        :return: a list of records (i.e. dictionaries) containing the data for the requested objects
        """

    @abc.abstractmethod
    def get_linked_records(self, cls_name: str, parent_cls_name: str, parent_id: int) -> list[dict[str, Any]]:
        """
        Abstract method enforcing the implementation of a method returning the list of records for all objects of class
         defined by class_name that are linked to object of class parent_class_name with id_ = parent_id

        :param cls_name: the class name of the objects to retrieve records for
        :param parent_cls_name: the class nae of the parent object
        :param parent_id: the id of the parent object
        :return: a list of records
        """

    @abc.abstractmethod
    def link(self, class1_name: str, id_1: int, class2_name: str, id_2: int) -> None:
        """
        Create a link between two domain-specific objects. There must be a direct link defined in Linker class,
        otherwise, the link cannot be created.

        :param class1_name: the class name of the first object to link
        :param id_1: the id of the first object to link
        :param class2_name: the class name of the second object to link
        :param id_2: the id of the second object to link
        """

    @abc.abstractmethod
    def unlink(self, class1_name: str, id_1: int, class2_name: str, id_2: int) -> None:
        """
        Remove the link between two domain-specific objects. There must be a direct link defined in Linker class,
        otherwise, the link cannot be removed.

        :param class1_name: the class name of the first object to unlink
        :param id_1: the id of the first object to unlink
        :param class2_name: the class name of the second object to unlink
        :param id_2: the id of the second object to unlink
        """
