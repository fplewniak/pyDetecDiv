#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Convenience file allowing import of several DAO classes in one line. It is thus possible to import DAOs as follows:
from pydetecdiv.persistence.sqlalchemy.orm.dao import FOVdao, ROIdao, DataDao, DatasetDao
The dso_dao_mapping dictionary maps the correspondence between domain-specific class names and DAO classes
"""
from pydetecdiv.persistence.sqlalchemy.orm.BoundingBoxDao import BoundingBoxDao
from pydetecdiv.persistence.sqlalchemy.orm.EntityDao import EntityDao
from pydetecdiv.persistence.sqlalchemy.orm.FOVdao import FOVdao
from pydetecdiv.persistence.sqlalchemy.orm.MaskDao import MaskDao
from pydetecdiv.persistence.sqlalchemy.orm.PointDao import PointDao
from pydetecdiv.persistence.sqlalchemy.orm.ROIdao import ROIdao
from pydetecdiv.persistence.sqlalchemy.orm.ExperimentDao import ExperimentDao
from pydetecdiv.persistence.sqlalchemy.orm.DataDao import DataDao
from pydetecdiv.persistence.sqlalchemy.orm.DatasetDao import DatasetDao
from pydetecdiv.persistence.sqlalchemy.orm.ImageResourceDao import ImageResourceDao
from pydetecdiv.persistence.sqlalchemy.orm.RunDao import RunDao

dso_dao_mapping = {
    'FOV': FOVdao,
    'ROI': ROIdao,
    'Experiment': ExperimentDao,
    'Dataset': DatasetDao,
    'Data': DataDao,
    'ImageResource': ImageResourceDao,
    'Run': RunDao,
    'Entity': EntityDao,
    'BoundingBox': BoundingBoxDao,
    'Point': PointDao,
    'Mask': MaskDao,
}
