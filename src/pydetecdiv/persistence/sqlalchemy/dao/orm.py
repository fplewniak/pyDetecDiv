#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
Classes of DAO accessing data in Tables, created by Table reflection.
These objects may not be necessary and may be removed in the near future unless decided otherwise
"""
from sqlalchemy.orm import registry
from pydetecdiv.persistence.sqlalchemy.dao.tables import Tables

mapper_registry = registry()
Base = mapper_registry.generate_base()
tables = Tables()


class FOV_DAO(Base):
    __table__ = tables.list['FOV']

    def __repr__(self):
        return f'FOV: {self.id!r}, {self.name!r}, {self.comments!r}'


class ROI_DAO(Base):
    __table__ = tables.list['ROI']

    def __repr__(self):
        return f'ROI: {self.id!r}, {self.name!r}'
