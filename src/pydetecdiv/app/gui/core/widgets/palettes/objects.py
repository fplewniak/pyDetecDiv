from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QTreeView, QMenu

from typing import TYPE_CHECKING

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.models.Trees import TreeDictModel, TreeItem

if TYPE_CHECKING:
    from pydetecdiv.app.gui.Windows import MainWindow



class ObjectItem(TreeItem):
    def __init__(self, graphics_item, data, parent=None):
        super().__init__(data, parent)
        self.object = graphics_item
        self.item_data = [graphics_item.data(0)]


class ObjectTreeView(QTreeView):
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

class ObjectTreeModel(TreeDictModel):
    def __init__(self, parent=None):
        self.object_dict = {}
        super().__init__([''], data=self.object_dict, parent=parent)
        PyDetecDiv.app.scene_modified.connect(self.update_model)

    def update_model(self, scene):
        self.beginResetModel()
        self.object_dict = scene.item_dict()
        self.root_item = TreeItem(self.root_item.item_data)
        self.setup_model_data(self.object_dict, self.root_item)
        self.endResetModel()

    def append_children(self, data, parent):
        """
        Append children to an arbitrary node represented by a dictionary. This method is called recursively to load the
        successive levels of nodes.

        :param data: the dictionary to load at this node
        :type data: dict
        :param parent: the internal node
        :type parent: TreeItem
        """
        for key, values in data.items():
            self.parents.append(TreeItem([key, ''], parent))
            parent.append_child(self.parents[-1])
            if isinstance(values, dict):
                self.append_children(values, self.parents[-1])
            else:
                for v in values:
                    self.parents[-1].append_child(ObjectItem(v, self.parents[-1]))


class ObjectTreePalette(QDockWidget):
    """

    """

    def __init__(self, parent: 'MainWindow'):
        super().__init__('Objects tree', parent)
        self.setObjectName('Objects tree')
        tree_view = ObjectTreeView()
        tree_view.setModel(ObjectTreeModel(parent=self))
        self.setWidget(tree_view)
