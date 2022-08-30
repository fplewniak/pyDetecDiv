#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
A class for the creation of tables that will serve to:
    1) create a persistence database if it does not exist, thus ensuring consistency across the persistence layer
    2) create the ORM classes in orm.py, but this role may be removed in the near future if the ORM is abandoned.
"""
from sqlalchemy import Table, Column, Integer, String, Time, DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import MetaData
from sqlalchemy.schema import Index
from sqlalchemy import text


class Tables():
    """
    A class defining the database tables for data access mapping. These tables are used to create the actual persistence
    database, specify SQL queries, and get results
    """
    def __init__(self):
        self.metadata_obj = MetaData()

        self.list = {
            'FOV': Table(
                'FOV',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('name', String, unique=True),
                Column('comments', String),
                Column('xsize', Integer, nullable=False, server_default=text('1000')),
                Column('ysize', Integer, nullable=False, server_default=text('1000')),
            ),

            'FOVdata': Table(
                'FOVdata',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('fov', ForeignKey('FOV.id'), nullable=False, index=True),
                Column('imagedata', ForeignKey('ImageData.id'), nullable=False, index=True),
                Index('fovdata_idx', 'fov', 'imagedata')
            ),

            'FOVprocess': Table(
                'FOVprocess',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('fov', ForeignKey('FOV.id'), nullable=False, index=True),
                Column('processing', ForeignKey('Processing.id'), nullable=False, index=True),
            ),

            'ROI': Table(
                'ROI',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('name', String, unique=True),
                Column('fov', ForeignKey('FOV.id'), nullable=False, index=True),
                Column('x0', Integer, nullable=False, server_default=text('0')),
                Column('y0', Integer, nullable=False, server_default=text('-1')),
                Column('x1', Integer, nullable=False, server_default=text('0')),
                Column('y1', Integer, nullable=False, server_default=text('-1')),
            ),

            'ROIdata': Table(
                'ROIdata',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('roi', ForeignKey('ROI.id'), nullable=False, index=True),
                Column('imagedata', ForeignKey('ImageData.id'), nullable=False, index=True),
                Index('roidata_idx', 'roi', 'imagedata')
            ),

            'ROIprocess': Table(
                'ROIprocess',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('roi', ForeignKey('ROI.id'), nullable=False, index=True),
                Column('processing', ForeignKey('Processing.id'), nullable=False, index=True),
            ),

            'Image': Table(
                'Image',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('z', Integer, nullable=False, server_default=text('0')),
                Column('t', Integer, nullable=False, server_default=text('0')),
                Column('xdrift', Integer, nullable=False, server_default=text('0')),
                Column('ydrift', Integer, nullable=False, server_default=text('0')),
                Column('data', ForeignKey('ImageData.id'), nullable=False, index=True),
                Index('zt_idx', 'z', 't')
            ),

            'ImageData': Table(
                'ImageData',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('name', String, server_default='original capture', unique=True),
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
            ),

            'FileResource': Table(
                'FileResource',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('locator', String, nullable=False, unique=True),
                Column('mimetype', String),
            ),

            'Results': Table(
                'Results',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('classification', String, nullable=False, index=True),
                Column('freetext', String),
            ),

            'ResultData': Table(
                'ResultData',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('results', ForeignKey('Results.id'), nullable=False, index=True),
                Column('resource', ForeignKey('FileResource.id'), nullable=False, index=True),
                Column('path', String, nullable=False),
                Column('mimetype', String),
            ),

            'ResultImageData': Table(
                'ResultImageData',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('results', ForeignKey('Results.id'), nullable=False, index=True),
                Column('imagedata', ForeignKey('ImageData.id'), nullable=False, index=True),
            ),

            'Processor': Table(
                'Processor',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('name', String, nullable=False, unique=True),
                Column('type', String, nullable=False),
                Column('description', String, ),
            ),

            'Processing': Table(
                'Processing',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('processor', ForeignKey('Processor.id'), nullable=False, index=True),
                Column('parameters', String),
            ),

            'History': Table(
                'History',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('time', DateTime, index=True),
            ),

            'HistoryFOVprocess': Table(
                'HistoryFOVprocess',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('history', ForeignKey('History.id'), nullable=False, index=True),
                Column('fovprocess', ForeignKey('FOVprocess.id'), nullable=False, index=True),
                Column('results', ForeignKey('Results.id'), nullable=False, index=True),
                Index('history_fov_idx', 'history', 'fovprocess', 'results')
            ),

            'HistoryROIprocess': Table(
                'HistoryROIprocess',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('history', ForeignKey('History.id'), nullable=False, index=True),
                Column('roiprocess', ForeignKey('ROIprocess.id'), nullable=False, index=True),
                Column('results', ForeignKey('Results.id'), nullable=False, index=True),
                Index('history_roi_idx', 'history', 'roiprocess', 'results')
            ),

            'Dataset': Table(
                'Dataset',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('name', String, nullable=False, unique=True),
            ),

            'ImageDataset': Table(
                'ImageDataset',
                self.metadata_obj,
                Column('id', Integer, primary_key=True, autoincrement='auto'),
                Column('dataset', ForeignKey('Dataset.id'), nullable=False, index=True),
                Column('image', ForeignKey('Image.id'), nullable=False, index=True),
                Index('image_dataset_idx', 'dataset', 'image')
            )
        }

    def columns(self, table: str = None):
        """
        Gets columns of a table designated by its name for simpler access to results
        :param table: the table name
        :return: the requested columns from the table object
        """
        return self.list[table].c
