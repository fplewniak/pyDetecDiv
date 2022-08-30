#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
A class for the creation of tables that will serve to:
    1) create a persistence if it does not exist, thus ensuring consistency between the persistence structure and ORM objects
    2) create the ORM classes in orm.py
"""
from sqlalchemy import Table, Column, Integer, String, Time, DateTime
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
            Column('name', String, unique=True),
            Column('comments', String),
            Column('xsize', Integer, nullable=False, server_default=text('1000')),
            Column('ysize', Integer, nullable=False, server_default=text('1000')),
        )

        self.fov_data = Table(
            'FOVdata',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('fov', ForeignKey('FOV.id'), nullable=False, index=True),
            Column('imagedata', ForeignKey('ImageData.id'), nullable=False, index=True),
            Index('fovdata_idx', 'fov', 'imagedata')
        )

        self.fov_process = Table(
            'FOVprocess',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('fov', ForeignKey('FOV.id'), nullable=False, index=True),
            Column('processing', ForeignKey('Processing.id'), nullable=False, index=True),
        )

        self.roi = Table(
            'ROI',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('name', String, unique=True),
            Column('fov', ForeignKey('FOV.id'), nullable=False, index=True),
            Column('x0', Integer, nullable=False, server_default=text('0')),
            Column('y0', Integer, nullable=False, server_default=text('-1')),
            Column('x1', Integer, nullable=False, server_default=text('0')),
            Column('y1', Integer, nullable=False, server_default=text('-1')),
        )

        self.roi_data = Table(
            'ROIdata',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('roi', ForeignKey('ROI.id'), nullable=False, index=True),
            Column('imagedata', ForeignKey('ImageData.id'), nullable=False, index=True),
            Index('roidata_idx', 'roi', 'imagedata')
        )

        self.roi_process = Table(
            'ROIprocess',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('roi', ForeignKey('ROI.id'), nullable=False, index=True),
            Column('processing', ForeignKey('Processing.id'), nullable=False, index=True),
        )

        self.image = Table(
            'Image',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('z', Integer, nullable=False, server_default=text('0')),
            Column('t', Integer, nullable=False, server_default=text('0')),
            Column('xdrift', Integer, nullable=False, server_default=text('0')),
            Column('ydrift', Integer, nullable=False, server_default=text('0')),
            Column('data', ForeignKey('ImageData.id'), nullable=False, index=True),
            Index('zt_idx', 'z', 't')
        )

        self.image_data = Table(
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
        )

        self.file_resource = Table(
            'FileResource',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('locator', String, nullable=False, unique=True),
            Column('mimetype', String),
        )

        self.results = Table(
            'Results',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('classification', String, nullable=False, index=True),
            Column('freetext', String),
        )

        self.result_data = Table(
            'ResultData',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('results', ForeignKey('Results.id'), nullable=False, index=True),
            Column('resource', ForeignKey('FileResource.id'), nullable=False, index=True),
            Column('path', String, nullable=False),
            Column('mimetype', String),
        )

        self.result_image_data = Table(
            'ResultImageData',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('results', ForeignKey('Results.id'), nullable=False, index=True),
            Column('imagedata', ForeignKey('ImageData.id'), nullable=False, index=True),
        )

        self.processor = Table(
            'Processor',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('name', String, nullable=False, unique=True),
            Column('type', String, nullable=False),
            Column('description', String, ),
        )

        self.processing = Table(
            'Processing',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('processor', ForeignKey('Processor.id'), nullable=False, index=True),
            Column('parameters', String),
        )

        self.history = Table(
            'History',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('time', DateTime, index=True),
        )

        self.history_fov_process = Table(
            'HistoryFOVprocess',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('history', ForeignKey('History.id'), nullable=False, index=True),
            Column('fovprocess', ForeignKey('FOVprocess.id'), nullable=False, index=True),
            Column('results', ForeignKey('Results.id'), nullable=False, index=True),
            Index('history_fov_idx', 'history', 'fovprocess', 'results')
        )

        self.history_roi_process = Table(
            'HistoryROIprocess',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('history', ForeignKey('History.id'), nullable=False, index=True),
            Column('roiprocess', ForeignKey('ROIprocess.id'), nullable=False, index=True),
            Column('results', ForeignKey('Results.id'), nullable=False, index=True),
            Index('history_roi_idx', 'history', 'roiprocess', 'results')
        )

        self.dataset = Table(
            'Dataset',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('name', String, nullable=False, unique=True),
        )

        self.image_dataset = Table(
            'ImageDataset',
            self.metadata_obj,
            Column('id', Integer, primary_key=True, autoincrement='auto'),
            Column('dataset', ForeignKey('Dataset.id'), nullable=False, index=True),
            Column('image', ForeignKey('Image.id'), nullable=False, index=True),
            Index('image_dataset_idx', 'dataset', 'image')
        )
