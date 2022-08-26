#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
Classes of DAO accessing data in Tables, created by Table reflection
"""
from sqlalchemy.orm import registry
from pydetecdiv.database.dao.sqlalchemy.tables import Tables

mapper_registry = registry()
Base = mapper_registry.generate_base()
tables = Tables()


class FOV_DAO(Base):
    __table__ = tables.fov

    #def __repr__(self):
    #    return f'FOV: {self.id!r}, {self.name!r}, {self.comments!r}'


class ROI_DAO(Base):
    __table__ = tables.roi

    def __repr__(self):
        return f'ROI: {self.id!r}, {self.name!r}'

