#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
Creation of global mapper_registry and Base class for database access.
Main DAO class for accessing data in SQL Tables. Subclasses are responsible for providing the domain layer with lists
# of compatible records for the creation of domain-specific objects.
"""
from typing import Any

from sqlalchemy.orm import registry
from sqlalchemy.orm.session import Session
from sqlalchemy import update, insert

mapper_registry = registry()
Base = mapper_registry.generate_base()


class DAO:
    """
    Data Access Object class defining methods common to all DAOs. This class is not meant to be used directly.
    Actual DAOs should inherit of this class first in order to inherit the __init__ method.
    """
    id_ = None
    exclude = []
    translate = {}

    def __init__(self, session: Session):
        self.session = session

    def insert(self, rec: dict[str, Any]) -> int:
        """
        Inserts data in SQL database for a newly created object

        :param rec: the record representing the object
        :return: the primary key of the newly created object
        """
        record = self.translate_record(rec, self.exclude, self.translate)
        stmt = (
            insert(self.__class__)
            .values(**record)
        )
        primary_key = self.session.execute(stmt).inserted_primary_key[0]
        # primary_key = self.session.execute(Insert(self.__class__).values(record)).inserted_primary_key[0]
        # with self.session() as session:
        #     primary_key = session.execute(Insert(self.__class__).values(record)).inserted_primary_key[0]
        #     session.commit()
        return primary_key

    def update(self, rec: dict[str, Any]) -> int:
        """
        Updates data in SQL database for the object corresponding to the record, which should contain the id of the
        modified object

        :param rec: the record representing the object
        :return: the primary key of the updated object
        """
        id_ = rec['id_']
        record = self.translate_record(rec, self.exclude, self.translate)
        # self.session.execute(Update(self.__class__, whereclause=self.__class__.id_ == id_).values(record))
        stmt = (
            update(self.__class__)
            .where(self.__class__.id_ == id_)
            .values(**record)
        )
        self.session.execute(stmt)
        # self.session.commit()
        return id_

    def get_records(self, where_clause) -> list[property]:
        """
        A method to get from the SQL database, all records verifying the specified where clause

        **Example of use:**
        ``roi_records = roidao.get_records((ROIdao.fov == FOVdao.id_) & (FOVdao.name == 'fov1'))``
        retrieves all ROI records associated with FOV whose name is 'fov1'

        :param where_clause: the selection 'where clause' that can be specified using DAO classes or tables
        :return: a list of records as dictionaries
        """
        dao_list = self.session.query(self.__class__).where(where_clause)
        return [obj.record for obj in dao_list]

    @staticmethod
    def translate_record(record: dict, exclude: list[str], translate: dict) -> dict:
        """
        The actual translation engine reading the record fields and translating or excluding them if they occur in the
        translate or exclude variables

        :param record: the record to be translated
        :param exclude: a list of fields that must not be passed to the SQL engine
        :param translate: a list of fields that must be translated and the corresponding columns
        :return: the translated record
        """
        rec = {}
        for key in record:
            if key in exclude:
                continue
            if key in translate:
                if isinstance(translate[key], dict):
                    rec |= record[key]
                elif isinstance(translate[key], (list, tuple)):
                    for translated_key, value in zip(translate[key], record[key]):
                        rec[translated_key] = value
                else:
                    rec[translate[key]] = record[key]
            else:
                rec[key] = record[key]
        return rec

    @property
    def record(self) -> dict:
        """
        Template for method converting a DAO row dictionary into a DSO record

        :return: a DSO record
        """
        raise NotImplementedError('Call to a record() method that is not implemented')
