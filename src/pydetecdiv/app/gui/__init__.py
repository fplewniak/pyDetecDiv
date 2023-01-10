#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import dearpygui.dearpygui as dpg
from pydetecdiv.domain.Project import Project


class GenericWidget:
    def __init__(self, tag, **kwargs):
        self.tag = tag

    def show_hide(self):
        dpg.configure_item(self.tag, show=not dpg.get_item_configuration(self.tag)['show'])

    def set_source(self, source=None):
        dpg.configure_item(self.tag, source=source)


class ObjectPool:
    def __init__(self):
        self.pool = {'Project': None}

    @property
    def project(self):
        return self.pool['Project']

    def close_project(self):
        if self.project is not None:
            self.project.repository.close()
            self.pool['Project'] = None
        return self

    def set_project(self, dbname):
        self.pool['Project'] = Project(dbname)
        return self

    def add_object(self, tag, obj):
        self.pool[tag] = obj


object_pool = ObjectPool()
