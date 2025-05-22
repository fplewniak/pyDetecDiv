# #  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
# #  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
#
# """
# A class for the creation of classes that was serving to:
#     1) create a persistence database if it does not exist, thus ensuring consistency across the persistence layer
#     2) create the ORM classes
# However, since switching to full-ORM, this class is not needed any more. It is kept for the moment only for memory
# until all the corresponding ORM classes have been implemented.
# """
# from sqlalchemy import Table, Column, Integer, String, DateTime
# from sqlalchemy import ForeignKey
# from sqlalchemy import MetaData
# from sqlalchemy.schema import Index
#
#
# class Tables:
#     """
#     A class defining the database classes for data access mapping. These classes are used to create the actual
#     persistence database, specify SQL queries, and get results
#     """
#     def __init__(self):
#         self.metadata_obj = MetaData()
#
#         self.list = {
#             'FOVprocess': Table(
#                 'FOVprocess',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('fov', ForeignKey('FOV.id'), nullable=False, index=True),
#                 Column('processing', ForeignKey('Processing.id'), nullable=False, index=True),
#             ),
#
#             'ROIprocess': Table(
#                 'ROIprocess',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('roi', ForeignKey('ROI.id'), nullable=False, index=True),
#                 Column('processing', ForeignKey('Processing.id'), nullable=False, index=True),
#             ),
#
#             'FileResource': Table(
#                 'FileResource',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('locator', String, nullable=False, unique=True),
#                 Column('mimetype', String),
#             ),
#
#             'Results': Table(
#                 'Results',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('classification', String, nullable=False, index=True),
#                 Column('freetext', String),
#             ),
#
#             'ResultData': Table(
#                 'ResultData',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('results', ForeignKey('Results.id'), nullable=False, index=True),
#                 Column('resource', ForeignKey('FileResource.id'), nullable=False, index=True),
#                 Column('path', String, nullable=False),
#                 Column('mimetype', String),
#             ),
#
#             'ResultImageData': Table(
#                 'ResultImageData',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('results', ForeignKey('Results.id'), nullable=False, index=True),
#                 Column('imagedata', ForeignKey('ImageData.id'), nullable=False, index=True),
#             ),
#
#             'Processor': Table(
#                 'Processor',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('name', String, nullable=False, unique=True),
#                 Column('type', String, nullable=False),
#                 Column('description', String, ),
#             ),
#
#             'Processing': Table(
#                 'Processing',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('processor', ForeignKey('Processor.id'), nullable=False, index=True),
#                 Column('parameters', String),
#             ),
#
#             'History': Table(
#                 'History',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('time', DateTime, index=True),
#             ),
#
#             'HistoryFOVprocess': Table(
#                 'HistoryFOVprocess',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('history', ForeignKey('History.id'), nullable=False, index=True),
#                 Column('fovprocess', ForeignKey('FOVprocess.id'), nullable=False, index=True),
#                 Column('results', ForeignKey('Results.id'), nullable=False, index=True),
#                 Index('history_fov_idx', 'history', 'fovprocess', 'results')
#             ),
#
#             'HistoryROIprocess': Table(
#                 'HistoryROIprocess',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('history', ForeignKey('History.id'), nullable=False, index=True),
#                 Column('roiprocess', ForeignKey('ROIprocess.id'), nullable=False, index=True),
#                 Column('results', ForeignKey('Results.id'), nullable=False, index=True),
#                 Index('history_roi_idx', 'history', 'roiprocess', 'results')
#             ),
#
#             'Dataset': Table(
#                 'Dataset',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('name', String, nullable=False, unique=True),
#             ),
#
#             'ImageDataset': Table(
#                 'ImageDataset',
#                 self.metadata_obj,
#                 Column('id', Integer, primary_key=True, autoincrement='auto'),
#                 Column('dataset', ForeignKey('Dataset.id'), nullable=False, index=True),
#                 Column('image', ForeignKey('Image.id'), nullable=False, index=True),
#                 Index('image_dataset_idx', 'dataset', 'image')
#             )
#         }
