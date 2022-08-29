#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
A class for the creation of tables that will serve to:
    1) create a persistence if it does not exist, thus ensuring consistency between the persistence structure and ORM objects
    2) create the ORM classes in orm.py
"""
from sqlalchemy import Table, Column, Integer, String, Time
from sqlalchemy import ForeignKey
from sqlalchemy import MetaData
from sqlalchemy.schema import UniqueConstraint, Index
from sqlalchemy import text

class Tables():
    def __init__(self):
        self.metadata_obj = MetaData()
        self.fov = Table(
            'FOV',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('name', String),
            Column('comments', String),
            Column('xsize', Integer, nullable=False, server_default=text('1000')),
            Column('ysize', Integer, nullable=False, server_default=text('1000')),
            UniqueConstraint('id'),
            UniqueConstraint('name'),
        )

        self.fovdata = Table(
            'FOVdata',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('fov', ForeignKey('FOV.id'), nullable=False, index=True),
            Column('imagedata', ForeignKey('ImageData.id'), nullable=False, index=True),
            UniqueConstraint('id'),
            Index('fovdata_idx', 'fov', 'imagedata')
        )

        self.roi = Table(
            'ROI',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('name', String),
            Column('fov', ForeignKey('FOV.id'), nullable=False, index=True),
            Column('x0', Integer, nullable=False, server_default=text('0')),
            Column('y0', Integer, nullable=False, server_default=text('-1')),
            Column('x1', Integer, nullable=False, server_default=text('0')),
            Column('y1', Integer, nullable=False, server_default=text('-1')),
            UniqueConstraint('id'),
            UniqueConstraint('name'),
        )

        self.roidata = Table(
            'ROIdata',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('roi', ForeignKey('ROI.id'), nullable=False, index=True),
            Column('imagedata', ForeignKey('ImageData.id'), nullable=False, index=True),
            UniqueConstraint('id'),
            Index('roidata_idx', 'roi', 'imagedata')
        )

        self.imagedata = Table(
            'ImageData',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('name', String, server_default='original capture'),
            Column('channel', Integer, nullable=False, server_default=text('0')),
            Column('x0', Integer, nullable=False, server_default=text('0')),
            Column('y0', Integer, nullable=False, server_default=text('-1')),
            Column('x1', Integer, nullable=False, server_default=text('0')),
            Column('y1', Integer, nullable=False, server_default=text('-1')),
            Column('zsize', Integer, nullable=False, server_default=text('1')),
            Column('tsize', Integer, nullable=False, server_default=text('1')),
            Column('interval', Time, nullable=False, server_default=text('0')),
            Column('orderdims', String, nullable=False, server_default='xyzct'),
            Column('resource', ForeignKey('FileResource.id'), nullable=False, index=True),
            Column('path', String, nullable=False),
            Column('mimetype', String),
            UniqueConstraint('id'),
            UniqueConstraint('name'),
        )

        self.fileresource = Table(
            'FileResource',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('locator', String, nullable=False),
            Column('mimetype', String),
            UniqueConstraint('id'),
            UniqueConstraint('locator'),
        )
