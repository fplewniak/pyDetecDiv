#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Access to ROI data
"""
from typing import Any

from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.types import JSON
from pydetecdiv.persistence.sqlalchemy.orm.main import DAO, Base


class RunDao(DAO, Base):
    """
    DAO class for access to Run records from the SQL database
    """
    __tablename__ = 'run'
    exclude = ['id_']
    translate = {}

    id_ = Column(Integer, primary_key=True, autoincrement='auto')
    uuid = Column(String(36),)
    tool_name = Column(String, nullable=False,)
    tool_version = Column(String, nullable=False,)
    is_plugin = Column(Boolean, nullable=False)
    command = Column(String)
    parameters = Column(JSON)
    key_val = Column(JSON)

    @property
    def record(self) -> dict[str, Any]:
        """
        A method creating a record dictionary from a dataset row dictionary. This method is used to convert the SQL
        table columns into the dataset record fields expected by the domain layer

        :return: a Run record as a dictionary with keys() appropriate for handling by the domain layer
        :rtype: dict
        """
        return {'id_': self.id_,
                'uuid': self.uuid,
                'tool_name': self.tool_name,
                'tool_version': self.tool_version,
                'is_plugin': self.is_plugin,
                'command': self.command,
                'parameters': self.parameters,
                'key_val': self.key_val,
                }
