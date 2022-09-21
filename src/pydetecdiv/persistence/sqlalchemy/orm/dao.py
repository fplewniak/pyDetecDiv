#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Convenience file allowing import of several DAO classes in one line. It is thus possible to import DAOs as follows:
from pydetecdiv.persistence.sqlalchemy.orm.dao import FOVdao, ROIdao, ImageDataDao, FileResourceDao
The dso_dao_mapping dictionary maps the correspondence between domain-specific class names and DAO classes
"""
from pydetecdiv.persistence.sqlalchemy.orm.FOVdao import FOVdao
from pydetecdiv.persistence.sqlalchemy.orm.ROIdao import ROIdao
from pydetecdiv.persistence.sqlalchemy.orm.ImageDataDao import ImageDataDao
from pydetecdiv.persistence.sqlalchemy.orm.FileResourceDao import FileResourceDao

dso_dao_mapping = {
        'FOV': FOVdao,
        'ROI': ROIdao,
        'ImageData': ImageDataDao,
        'FileResource': FileResourceDao
    }
