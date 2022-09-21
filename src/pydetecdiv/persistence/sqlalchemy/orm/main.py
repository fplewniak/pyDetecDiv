#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
Creation of global mapper_registry and Base class for database access.
Main DAO class for accessing data in SQL Tables. Subclasses are responsible for providing the domain layer with lists
of compatible records for the creation of domain-specific objects.
"""
from sqlalchemy.orm import registry
from sqlalchemy.sql.expression import Insert, Update

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

    def __init__(self, session_maker):
        self.session_maker = session_maker

    def insert(self, rec):
        """
        Inserts data in SQL database for a newly created object
        :param rec: the record representing the object
        :type rec: dict
        :return: the primary key of the newly created object
        :rtype: int
        """
        record = self.translate_record(rec, self.exclude, self.translate)
        with self.session_maker() as session:
            primary_key = session.execute(Insert(self.__class__).values(record)).inserted_primary_key[0]
            session.commit()
        return primary_key

    def update(self, rec):
        """
        Updates data in SQL database for the object corresponding to the record, which should contain the id of the
        modified object
        :param rec: the record representing the object
        :type rec: dict
        :return: the primary key of the updated object
        :rtype: int
        """
        id_ = rec['id_']
        record = self.translate_record(rec, self.exclude, self.translate)
        with self.session_maker() as session:
            session.execute(Update(self.__class__, whereclause=self.__class__.id_ == id_).values(record))
            session.commit()
        return id_

    def get_records(self, where_clause):
        """
        A method to get from the SQL database, all records verifying the specified where clause
        Example of use:
        roi_records = roidao.get_records((ROIdao.fov == FOVdao.id_) & (FOVdao.name == 'fov1')) retrieves all ROI records
        associated with FOV whose name is 'fov1'

        :param where_clause: the selection 'where clause' that can be specified using DAO classes or tables
        :type where_clause: a sqlachemy where clause defining the SQL selection query
        :return: a list of records as dictionaries
        :rtype: list of dict
        """
        with self.session_maker() as session:
            dao_list = session.query(self.__class__).where(where_clause)
        return [obj.record for obj in dao_list]

    @staticmethod
    def translate_record(record, exclude, translate):
        """
        The actual translation engine reading the record fields and translating or excluding them if they occur in the
        translate or exclude variables
        :param record: the record to be translated
        :type record: dict
        :param exclude: a list of fields that must not be passed to the SQL engine
        :type exclude: list
        :param translate: a list of fields that must be translated and the corresponding columns
        :type translate: dict
        :return: the translated record
        :rtype: dict
        """
        rec = {}
        for key in record:
            if key in exclude:
                continue
            if key in translate:
                if isinstance(translate[key], (list, tuple)):
                    for translated_key, value in zip(translate[key], record[key]):
                        rec[translated_key] = value
                else:
                    rec[translate[key]] = record[key]
            else:
                rec[key] = record[key]
        return rec

    @property
    def record(self):
        """
        Template for method converting a DAO row dictionary into a DSO record
        :return: a DSO record
        :rtype: dict
        """
        raise NotImplementedError('Call to a record() method that is not implemented')
