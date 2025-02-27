import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QTreeView, QMenu

from typing import TYPE_CHECKING

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.models.Trees import TreeDictModel, TreeItem, TreeModel

if TYPE_CHECKING:
    from pydetecdiv.app.gui.Windows import MainWindow


class SceneTreeItem(TreeItem):
    def __init__(self, graphics_item, data, parent=None):
        super().__init__(data, parent)
        self.object = graphics_item
        self.item_data = [graphics_item.data(0)]


class SceneTreeView(QTreeView):
    """
    A class expanding QTreeView with specific features to view objects in Image/Video viewer.
    """

    def contextMenuEvent(self, event):
        """
        The context menu for area manipulation

        :param event:
        """
        index = self.currentIndex()
        rect = self.visualRect(index)
        if index and rect.top() <= event.pos().y() <= rect.bottom():
            menu = QMenu()
            view_info = menu.addAction("View info")
            view_info.triggered.connect(self.view_info())
            menu.exec(self.viewport().mapToGlobal(event.pos()))

    def view_info(self):
        selection = self.currentIndex()
        print(selection.row())
        print(selection.internalPointer().item_data)
        print(self.model().data(selection, Qt.DisplayRole))


class SceneTreeModel(TreeModel):
    def __init__(self, top_items=None, parent=None):
        super().__init__([''], parent=parent)
        PyDetecDiv.app.scene_modified.connect(self.update_model)
        PyDetecDiv.app.graphic_item_deleted.connect(self.delete_graphic_item)
        PyDetecDiv.app.other_scene_in_focus.connect(self.reset_model)
        self.all_items = set()
        self.top_items = {
            item: TreeItem([item, None]) for item in top_items
            } if top_items is not None else {}
        for i in self.top_items.values():
            self.add_item(self.root_item, i)
        self.current_scene = None
        # self.layers_item = TreeItem(['layers'])
        # self.boxes_item = TreeItem(['boxes'])
        # self.points_item = TreeItem(['points'])
        # self.add_item(self.root_item, self.layers_item)
        # self.add_item(self.root_item, self.boxes_item)
        # self.add_item(self.root_item, self.points_item)

    def add_item(self, parent_item: TreeItem, new_item: TreeItem):
        parent_item.append_child(new_item)
        self.all_items.add(new_item)
        self.layoutChanged.emit()
        return new_item

    def delete_graphic_item(self, item):
        item_list = {i.data(0): i for i in self.all_items}
        if item in item_list:
            item_list[item].parent().child_items.remove(item_list[item])
            del (item_list[item])
            self.layoutChanged.emit()

    def reset_model(self, scene):
        self.all_items = set()
        for item in self.top_items.values():
            item.child_items = []
        # self.top_items = {
        #     'layers': TreeItem(['layers']),
        #     'boxes' : TreeItem(['boxes']),
        #     'points': TreeItem(['points']),
        #     }
        # for i in self.top_items.values():
        #     self.add_item(self.root_item, i)
        if scene is not None:
            self.update_model(scene)

    def update_model(self, scene):
        self.current_scene = scene
        if 'layers' in self.top_items:
            layers_set = set([i for i in scene.layers() if i.data(0) is not None])
            layer_items = set([i.data(1) for i in self.top_items['layers'].child_items])
            for item in layers_set.difference(layer_items):
                self.add_item(self.top_items['layers'], TreeItem([item.data(0), item]))

        if 'boxes' in self.top_items:
            boxes_set = set([i for i in scene.regions() if i.data(0) is not None])
            box_items = set([i.data(1) for i in self.top_items['boxes'].child_items])
            for item in boxes_set.difference(box_items):
                self.add_item(self.top_items['boxes'], TreeItem([item.data(0), item]))

        if 'points' in self.top_items:
            points_set = set([i for i in scene.points() if i.data(0) is not None])
            point_items = set([i.data(1) for i in self.top_items['points'].child_items])
            for item in points_set.difference(point_items):
                self.add_item(self.top_items['points'], TreeItem([item.data(0), item]))


# class ObjectTreeDictModel(TreeDictModel):
#     def __init__(self, parent=None):
#         self.object_dict = {}
#         super().__init__([''], data=self.object_dict, parent=parent)
#         PyDetecDiv.app.scene_modified.connect(self.update_model)
#
#     def update_model(self, scene):
#         self.beginResetModel()
#         self.object_dict = scene.item_dict()
#         self.root_item = TreeItem(self.root_item.item_data, None)
#         self.setup_model_data(self.object_dict)
#         self.endResetModel()
#
#     def append_children(self, data, parent):
#         """
#         Append children to an arbitrary node represented by a dictionary. This method is called recursively to load the
#         successive levels of nodes.
#
#         :param data: the dictionary to load at this node
#         :type data: dict
#         :param parent: the internal node
#         :type parent: TreeItem
#         """
#         for key, values in data.items():
#             self.parents.append(TreeItem([key, ''], parent))
#             parent.append_child(self.parents[-1])
#             if isinstance(values, dict):
#                 self.append_children(values, self.parents[-1])
#             else:
#                 for v in values:
#                     self.parents[-1].append_child(SceneTreeItem(v, self.parents[-1]))


class SceneTreePalette(QDockWidget):
    """

    """

    def __init__(self, parent: 'MainWindow'):
        super().__init__('Scene palette', parent)
        self.setObjectName('Scene palette')
        self.tree_view = SceneTreeView()
        self.tree_view.setModel(SceneTreeModel(top_items=['layers', 'boxes', 'points'], parent=self))
        self.tree_view.setHeaderHidden(True)
        self.tree_view.expandAll()
        self.setWidget(self.tree_view)

    def set_top_items(self, top_items=None):
        if top_items is None:
            top_items = ['layers', 'boxes', 'points']
        # self.tree_view.model().top_items = {item: TreeItem([item, None]) for item in top_items}
        self.tree_view.setModel(SceneTreeModel(top_items=top_items, parent=self))

    def reset(self):
        self.set_top_items()
