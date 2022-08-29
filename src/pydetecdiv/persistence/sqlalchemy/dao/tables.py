#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM

"""
A class for the creation of tables that will serve to:
    1) create a persistence if it does not exist, thus ensuring consistency between the persistence structure and ORM objects
    2) create the ORM classes in orm.py
"""
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy import ForeignKey
from sqlalchemy import MetaData

class Tables():
    def __init__(self):
        self.metadata_obj = MetaData()
        self.fov = Table(
            'FOV',
            self.metadata_obj,
            Column('id', Integer, primary_key=True),
            Column('name', String),
            Column('comments', String),
            Column('xsize', Integer),
            Column('ysize', Integer)
        )
        self.roi = Table(
            'ROI',
            self.metadata_obj,
            Column('id', Integer, primary_key=True),
            Column('name', String),
            Column('fov', ForeignKey('FOV.id'), nullable=False),
            Column('x0', Integer),
            Column('y0', Integer),
            Column('x1', Integer),
            Column('y1', Integer),
        )




